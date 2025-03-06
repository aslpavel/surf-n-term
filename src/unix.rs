//! Unix systems specific `Terminal` implementation.
use crate::common::{IOQueue, env_cfg};
use crate::decoder::KEYBOARD_LEVEL;
use crate::encoder::ColorDepth;
use crate::image::ImageHandlerKind;
use crate::{
    DecMode, ImageHandler,
    decoder::{Decoder, TTYEventDecoder},
    encoder::{Encoder, TTYEncoder},
    error::Error,
    image::DummyImageHandler,
    terminal::{
        Size, Terminal, TerminalCommand, TerminalEvent, TerminalSize, TerminalStats, TerminalWaker,
    },
};
use crate::{Position, RGBA, TerminalCaps};
use rustix::event::FdSetElement;
use signal_hook::{
    consts::{SIGINT, SIGQUIT, SIGTERM, SIGWINCH},
    iterator::{backend::SignalDelivery, exfiltrator::SignalOnly},
};
use std::collections::HashMap;
use std::io::ErrorKind;
use std::os::fd::{BorrowedFd, OwnedFd};
use std::{
    collections::{HashSet, VecDeque},
    fs::File,
    io::{BufWriter, Cursor, Read, Write},
    os::unix::{
        io::{AsFd, AsRawFd, RawFd},
        net::UnixStream,
    },
    path::Path,
    time::{Duration, Instant},
};

pub struct UnixTerminal {
    tty: Tty,
    encoder: TTYEncoder,
    write_queue: IOQueue,
    decoder: TTYEventDecoder,
    events_queue: VecDeque<TerminalEvent>,
    waker_read: UnixStream,
    waker: TerminalWaker,
    termios_saved: rustix::termios::Termios,
    signal_delivery: SignalDelivery<UnixStream, SignalOnly>,
    stats: TerminalStats,
    tee: Option<BufWriter<File>>,
    image_handler: Box<dyn ImageHandler + 'static>,
    capabilities: TerminalCaps,
    // if it is not None we are going to use escape sequence to detect
    // terminal size, otherwise ioctl is used.
    size: Option<TerminalSize>,
    poll: Poll,
}

impl UnixTerminal {
    /// Create new terminal by opening `/dev/tty` device.
    pub fn new() -> Result<Self, Error> {
        let tty_fd = rustix::fs::open(
            "/dev/tty",
            rustix::fs::OFlags::RDWR,
            rustix::fs::Mode::empty(),
        )
        .or_else(|error| {
            // LLDB is not creating /dev/tty for child processes
            tracing::error!(
                "[UnixTerminal.new] failed to open terminal at /dev/tty with error {error:?}"
            );
            tracing::error!("[UnixTerminal.new] trying to fallback back to /dev/stdin");
            rustix::fs::open(
                "/dev/stdin",
                rustix::fs::OFlags::RDWR,
                rustix::fs::Mode::empty(),
            )
        })?;
        Self::new_from_fd(tty_fd)
    }

    /// Open terminal by a given device path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        Self::new_from_fd(rustix::fs::open(
            path.as_ref(),
            rustix::fs::OFlags::RDWR,
            rustix::fs::Mode::empty(),
        )?)
    }

    /// Create new terminal from raw file descriptor pointing to a tty.
    pub fn new_from_fd(tty_fd: OwnedFd) -> Result<Self, Error> {
        let tty = Tty::new(tty_fd);
        tty.set_nonblocking(true)?;
        if !rustix::termios::isatty(&tty) {
            return Err(Error::NotATTY);
        }

        let mut poll = Poll::new();
        poll.register(PollEvent::new(&tty).with_readable(true))?;

        // switching terminal into a raw mode
        // [Entering Raw Mode](https://viewsourcecode.org/snaptoken/kilo/02.enteringRawMode.html)
        let termios_saved = rustix::termios::tcgetattr(&tty)?;
        let mut termios = termios_saved.clone();
        termios.make_raw();
        rustix::termios::tcsetattr(&tty, rustix::termios::OptionalActions::Flush, &termios)?;

        // signal delivery
        let (signal_read, signal_write) = UnixStream::pair()?;
        let signal_delivery = SignalDelivery::with_pipe(
            signal_read,
            signal_write,
            SignalOnly,
            [SIGWINCH, SIGTERM, SIGINT, SIGQUIT],
        )?;
        poll.register(PollEvent::new(signal_delivery.get_read()).with_readable(true))?;

        // self-pipe trick to implement waker
        let (waker_read, waker_write) = UnixStream::pair()?;
        waker_write.set_nonblocking(true)?;
        waker_read.set_nonblocking(true)?;
        let waker = TerminalWaker::new(move || {
            const WAKE: &[u8] = b"\x00";
            // use write syscall instead of locking so it would be safe to use in a signal handler
            match rustix::io::write(&waker_write, WAKE) {
                Ok(_) | Err(rustix::io::Errno::INTR | rustix::io::Errno::AGAIN) => Ok(()),
                Err(error) => Err(error.into()),
            }
        });
        poll.register(PollEvent::new(&waker_read).with_readable(true))?;

        let capabilities = TerminalCaps::default();
        let mut term = Self {
            tty,
            encoder: TTYEncoder::new(capabilities.clone()),
            write_queue: Default::default(),
            decoder: TTYEventDecoder::new(),
            events_queue: Default::default(),
            waker_read,
            waker,
            termios_saved,
            signal_delivery,
            stats: TerminalStats::new(),
            tee: None,
            image_handler: Box::new(DummyImageHandler),
            capabilities,
            size: None,
            poll,
        };

        capabilities_detect(&mut term)?;
        term.execute(TerminalCommand::KeyboardLevel(KEYBOARD_LEVEL))?;
        Ok(term)
    }

    /// Duplicate all output to specified tee file. Used for debugging.
    pub fn duplicate_output(&mut self, path: impl AsRef<Path>) -> Result<(), Error> {
        let file = File::create(path)?;
        self.tee = Some(BufWriter::new(file));
        Ok(())
    }

    /// Statistics collected by terminal.
    pub fn stats(&self) -> &TerminalStats {
        &self.stats
    }

    /// Get a reference an image handler
    pub fn image_handler(&mut self) -> &mut dyn ImageHandler {
        &mut self.image_handler
    }

    /// Determine terminal size with ioctl
    ///
    /// Some terminal emulators do not set pixel size, or if it goes through some
    /// kind of muxer (like `docker exec`) which might not set pixel size. So if this
    /// condition is detected we are falling back to escape sequence if it detected to
    /// work.
    fn size_ioctl(&self) -> Result<TerminalSize, Error> {
        let winsize = rustix::termios::tcgetwinsize(&self.tty)?;
        Ok(TerminalSize {
            cells: Size {
                height: winsize.ws_row as usize,
                width: winsize.ws_col as usize,
            },
            pixels: Size {
                height: winsize.ws_ypixel as usize,
                width: winsize.ws_xpixel as usize,
            },
        })
    }

    /// Close all descriptors free all the resources
    fn dispose(&mut self) -> Result<(), Error> {
        self.frames_drop();

        // flush currently queued output and submit the epilogue
        self.execute_many([
            TerminalCommand::Face(Default::default()),
            TerminalCommand::visible_cursor_set(true),
            TerminalCommand::DecModeSet {
                enable: false,
                mode: DecMode::MouseMotions,
            },
            TerminalCommand::DecModeSet {
                enable: false,
                mode: DecMode::MouseSGR,
            },
            TerminalCommand::DecModeSet {
                enable: false,
                mode: DecMode::MouseReport,
            },
            TerminalCommand::DecModeSet {
                enable: true,
                mode: DecMode::AutoWrap,
            },
            TerminalCommand::KeyboardLevel(0),
            // This one must be the last command as we use it as sync event,
            // which is supported by all terminals, and indicates that we handled
            // all pending events such as status report from kitty image protocol
            TerminalCommand::DeviceAttrs,
        ])
        .unwrap_or(()); // ignore write errors

        // wait for device attributes report or error
        loop {
            match self.poll(Some(Duration::from_secs(1))) {
                Err(_) | Ok(Some(TerminalEvent::DeviceAttrs(_)) | None) => break,
                _ => {}
            }
        }

        // disable signal handler
        self.signal_delivery.handle().close();

        // restore terminal settings
        rustix::termios::tcsetattr(
            &self.tty,
            rustix::termios::OptionalActions::Flush,
            &self.termios_saved,
        )?;

        Ok(())
    }
}

/// Fallback way to determine terminal size if it is detected to work
/// and ioctl is not.
const GET_TERM_SIZE: &[u8] = b"\x1b[18t\x1b[14t";

/// Detect and set terminal capabilities
fn capabilities_detect(term: &mut UnixTerminal) -> Result<(), Error> {
    if let Ok("linux") | Ok("dumb") = std::env::var("TERM").as_deref() {
        // do not try to query anything on dumb terminals
        tracing::warn!("[capabilities_detected] dumb terminal");
        term.capabilities.depth = ColorDepth::Gray;
        term.encoder = TTYEncoder::new(term.capabilities.clone());
        return Ok(());
    }
    let mut caps = TerminalCaps::default();
    if let Ok("truecolor") | Ok("24bit") = std::env::var("COLORTERM").as_deref() {
        caps.depth = ColorDepth::TrueColor;
    }

    // drain all pending events
    term.drain().count();
    // NOTE: using `write!` here instead of execute, to not accidentally use
    //       existing configuration from passed terminal.

    // 1x1 pixel kitty image (NOTE: it will be consumed by handler if it is already set)
    write!(term, "\x1b_Ga=q,i=31,s=1,v=1,f=24;AAAA\x1b\\")?;

    // OSC - Get default background color for transparent blending
    write!(term, "\x1b]11;?\x1b\\")?;

    // Set background color with SGR, and try to get it back to
    // detect true color support https://github.com/termstandard/colors
    let face_expected = "bg=#010203".parse()?;
    write!(term, "\x1b[00;48;2;1;2;3m")?; // change background
    write!(term, "\x1bP$qm\x1b\\")?; // FaceGet (DECRQSS with `m` descriptor)
    write!(term, "\x1b[00m")?; // reset current face

    // Detect terminal size
    // Some terminals return incomplete size info with ioctl
    term.write_all(GET_TERM_SIZE)?;

    // Detect kitty keyboard protocol support
    write!(term, "\x1b[?u")?;

    // DA1 - sync and sixel info
    // Device Attribute command is used as "sync" event, it is supported
    // by most terminals, at least in its basic form, so we expect to
    // receive a response to it. Which means it should go LAST
    term.execute(TerminalCommand::DeviceAttrs)?;

    let mut image_handlers = HashSet::new();
    let mut bg: Option<RGBA> = None;
    let mut size_escape = TerminalSize::default();
    loop {
        match term.poll(Some(Duration::from_secs(1)))? {
            Some(TerminalEvent::KittyImage { .. }) => {
                tracing::debug!("[capabilities_detected] kitty image protocol");
                image_handlers.insert(ImageHandlerKind::Kitty);
            }
            Some(TerminalEvent::Color { color, .. }) => {
                tracing::debug!("[capabilities_detected] background color: {:?}", color);
                bg.replace(color);
            }
            Some(TerminalEvent::FaceGet(face)) => {
                if face == face_expected {
                    tracing::debug!("[capabilities_detected] true color support");
                    caps.depth = ColorDepth::TrueColor;
                }
            }
            Some(TerminalEvent::DeviceAttrs(attrs)) => {
                // 4 - attribute indicates sixel support
                if attrs.contains(&4) {
                    tracing::debug!("[capabilities_detected] sixel image protocol");
                    image_handlers.insert(ImageHandlerKind::Sixel);
                }
                break; // this is last "sync" event
            }
            Some(TerminalEvent::Size(size)) => {
                size_escape = size;
            }
            Some(TerminalEvent::KeyboardLevel(_)) => {
                tracing::debug!("[capabilities_detected] kitty keyboard protocol");
                caps.kitty_keyboard = true;
            }
            Some(event) => {
                tracing::warn!("[capabilities_detected] unexpected event: {:?}", event);
                continue;
            }
            None => break,
        }
    }

    // drain terminal
    term.drain().count();

    // color depth
    if let Some(depth) = env_cfg::<ColorDepth>("depth") {
        caps.depth = depth;
    }

    // term size interface
    let size_ioctl = term.size_ioctl()?;
    if size_ioctl.pixels.is_empty() && !size_escape.pixels.is_empty() {
        tracing::warn!("[capabilities_detected] fallback to escape sequence for term size");
        term.size = Some(size_escape);
    }

    // image handler
    let image_handler = env_cfg::<ImageHandlerKind>("image")
        .or_else(|| image_handlers.get(&ImageHandlerKind::Kitty).copied())
        .or_else(|| image_handlers.get(&ImageHandlerKind::Sixel).copied())
        .unwrap_or(ImageHandlerKind::Dummy)
        .into_image_handler(bg);

    // glyph support
    caps.glyphs = matches!(
        image_handler.kind(),
        ImageHandlerKind::Kitty | ImageHandlerKind::Sixel
    ) && !term.size()?.pixels.is_empty();

    // update terminal
    tracing::info!("[capabilities_detected] {:?}", caps);
    term.encoder = TTYEncoder::new(caps.clone());
    term.image_handler = image_handler;
    term.capabilities = caps;

    Ok(())
}

impl std::ops::Drop for UnixTerminal {
    fn drop(&mut self) {
        self.dispose().unwrap_or(())
    }
}

impl Write for UnixTerminal {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.write_queue.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.write_queue.flush()
    }
}

impl Terminal for UnixTerminal {
    #[tracing::instrument(name="[UnixTerminal.poll]", level="trace", skip_all, fields(?timeout))]
    fn poll(&mut self, timeout: Option<Duration>) -> Result<Option<TerminalEvent>, Error> {
        self.write_queue.flush()?;

        let mut first_loop = true;
        let timeout_instant = timeout.map(|dur| Instant::now() + dur);
        while !self.write_queue.is_empty() || self.events_queue.is_empty() {
            // process timeout
            let delay = match timeout_instant {
                Some(timeout_instant) => {
                    let now = Instant::now();
                    if timeout_instant < now {
                        if first_loop {
                            // execute first loop even if timeout is 0
                            Some(Duration::new(0, 0))
                        } else {
                            break;
                        }
                    } else {
                        Some(timeout_instant - now)
                    }
                }
                None => None,
            };

            let tty_write = PollEvent::new(&self.tty).with_writable(!self.write_queue.is_empty());
            self.poll.register(tty_write)?;
            let (waker, signal, tty) = match self.poll.wait(delay) {
                Ok(events) => {
                    tracing::trace!(count = events.len(), "[UnixTerminal.poll] events");
                    (
                        events.get(&self.waker_read),
                        events.get(self.signal_delivery.get_read()),
                        events.get(&self.tty),
                    )
                }
                Err(error) => {
                    if matches!(error.kind(), ErrorKind::Interrupted | ErrorKind::WouldBlock) {
                        continue;
                    } else {
                        return Err(error.into());
                    }
                }
            };

            // process pending output
            if tty.is_writable() {
                let tee = self.tee.as_mut();
                let send = self.write_queue.consume_with(|slice| {
                    let size = guard_io(self.tty.write(slice), 0)?;
                    tee.map(|tee| tee.write(&slice[..size])).transpose()?;
                    Ok::<_, Error>(size)
                })?;
                self.stats.send += send;
            }

            // process signals
            if signal.is_readable() {
                for signal in self.signal_delivery.pending() {
                    match signal {
                        SIGWINCH => {
                            if self.size.is_none() {
                                self.events_queue
                                    .push_back(TerminalEvent::Resize(self.size()?));
                            } else {
                                self.write_all(GET_TERM_SIZE)?;
                            }
                        }
                        SIGTERM | SIGINT | SIGQUIT => {
                            return Err(Error::Quit);
                        }
                        _ => {}
                    }
                }
            }

            // process waker
            if waker.is_readable() {
                let mut buf = [0u8; 1024];
                if guard_io(self.waker_read.read(&mut buf), 0)? != 0 {
                    self.events_queue.push_back(TerminalEvent::Wake);
                }
            }

            // process pending input
            if tty.is_readable() {
                let mut buf = [0u8; 1024];
                let recv = guard_io(self.tty.read(&mut buf), 0)?;
                if recv == 0 {
                    return Err(Error::Quit);
                }
                self.stats.recv += recv;
                tracing::trace!(
                    size = %recv,
                    data = format!("{}", buf[..recv].escape_ascii()),
                    "[UnixTerminal.poll] received:"
                );
                // parse events
                let mut read_queue = Cursor::new(&buf[..recv]);
                while let Some(event) = self.decoder.decode(&mut read_queue)? {
                    if let TerminalEvent::Size(size) = event {
                        // we are using escape sequence to determine terminal resize
                        if let Some(term_size) = self.size.as_mut() {
                            *term_size = size;
                            self.events_queue.push_back(TerminalEvent::Resize(size));
                        }
                    }
                    if !self.image_handler.handle(&mut self.write_queue, &event)? {
                        self.events_queue.push_back(event)
                    }
                }
            }

            // indicate that first loop was executed
            first_loop = false;
        }

        Ok(self.events_queue.pop_front())
    }

    fn execute(&mut self, cmd: TerminalCommand) -> Result<(), Error> {
        tracing::trace!(?cmd, "[UnixTerminal.execute]");
        match cmd {
            TerminalCommand::Image(img, pos) => {
                self.image_handler.draw(&mut self.write_queue, &img, pos)
            }
            TerminalCommand::ImageErase(img, pos) => {
                self.image_handler.erase(&mut self.write_queue, &img, pos)
            }
            cmd => self.encoder.encode(&mut self.write_queue, cmd),
        }
    }

    fn size(&self) -> Result<TerminalSize, Error> {
        match self.size {
            Some(size) => Ok(size),
            None => self.size_ioctl(),
        }
    }

    fn position(&mut self) -> Result<Position, Error> {
        let mut queue = Vec::new();
        self.execute(TerminalCommand::CursorGet)?;
        self.execute(TerminalCommand::DeviceAttrs)?; // sync event
        let mut pos = Position::origin();
        while let Some(event) = self.poll(None)? {
            match event {
                TerminalEvent::DeviceAttrs(..) => {
                    self.events_queue.extend(queue);
                    return Ok(pos);
                }
                TerminalEvent::CursorPosition(term_pos) => {
                    pos = term_pos;
                }
                event => queue.push(event),
            }
        }
        Ok(pos)
    }

    fn waker(&self) -> TerminalWaker {
        self.waker.clone()
    }

    fn frames_pending(&self) -> usize {
        self.write_queue.chunks_count()
    }

    fn frames_drop(&mut self) {
        self.write_queue.clear_but_last()
    }

    fn dyn_ref(&mut self) -> &mut dyn Terminal {
        self
    }

    fn capabilities(&self) -> &TerminalCaps {
        &self.capabilities
    }
}

fn guard_io<T>(result: Result<T, std::io::Error>, otherwise: T) -> Result<T, std::io::Error> {
    use std::io::ErrorKind::*;
    match result {
        Err(error) if error.kind() == WouldBlock || error.kind() == Interrupted => Ok(otherwise),
        _ => result,
    }
}

/// TTY Handle
#[derive(Debug)]
struct Tty {
    fd: OwnedFd,
}

impl Tty {
    pub fn new(fd: OwnedFd) -> Self {
        Self { fd }
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> Result<(), Error> {
        let mut flags = rustix::fs::fcntl_getfl(self)?;
        flags.set(rustix::fs::OFlags::NONBLOCK, nonblocking);
        rustix::fs::fcntl_setfl(self, flags)?;
        Ok(())
    }
}

impl AsRawFd for Tty {
    fn as_raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }
}

impl AsFd for Tty {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }
}

impl Write for Tty {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        rustix::io::write(self, buf).map_err(std::io::Error::from)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Read for Tty {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        rustix::io::read(self, buf).map_err(std::io::Error::from)
    }
}

// We have to use `select` in the implementation as under MacOS the only way to
// poll `/dev/tty` is to use select. `poll` for example would return `POLLNVAL`
#[derive(Default)]
struct Poll {
    registred: HashMap<RawFd, PollEvent>,
    fd_count: usize,
    fd_max: RawFd,
    read_set: Vec<rustix::event::FdSetElement>,
    write_set: Vec<rustix::event::FdSetElement>,
    matched: HashMap<RawFd, PollEvent>,
}

impl Poll {
    fn new() -> Self {
        Default::default()
    }

    fn register(&mut self, event: PollEvent) -> Result<(), std::io::Error> {
        let event_registred = self.registred.entry(event.fd).or_insert_with(|| {
            self.fd_count += 1;
            self.fd_max = self.fd_max.max(event.fd);
            let set_len = rustix::event::fd_set_num_elements(self.fd_count, self.fd_max + 1);
            if self.read_set.len() != set_len {
                self.read_set.resize_with(set_len, FdSetElement::default);
                self.write_set.resize_with(set_len, FdSetElement::default);
            }
            PollEvent::from_fd(event.fd)
        });
        if event.readable.is_some() {
            event_registred.readable = event.readable;
        }
        if event.writable.is_some() {
            event_registred.writable = event.writable;
        }
        Ok(())
    }

    fn wait(&mut self, timeout: Option<Duration>) -> Result<PollEvents<'_>, std::io::Error> {
        // setup sets
        self.read_set.fill(Default::default());
        self.write_set.fill(Default::default());
        self.registred.retain(|_, event| {
            if event.is_readable() {
                rustix::event::fd_set_insert(&mut self.read_set, event.fd);
            }
            if event.is_writable() {
                rustix::event::fd_set_insert(&mut self.write_set, event.fd);
            }
            !event.is_unset()
        });

        // select
        let timeout = timeout.map(|dur| rustix::fs::Timespec {
            tv_sec: dur.as_secs() as rustix::fs::Secs,
            tv_nsec: dur.subsec_nanos() as rustix::fs::Nsecs,
        });
        unsafe {
            rustix::event::select(
                self.fd_max + 1,
                Some(&mut self.read_set),
                Some(&mut self.write_set),
                None,
                timeout.as_ref(),
            )?;
        }

        // collect result
        self.matched.clear();
        for fd in rustix::event::FdSetIter::new(&self.read_set) {
            let event = self
                .matched
                .entry(fd)
                .or_insert_with(|| PollEvent::from_fd(fd));
            event.readable = Some(true);
        }
        for fd in rustix::event::FdSetIter::new(&self.write_set) {
            let event = self
                .matched
                .entry(fd)
                .or_insert_with(|| PollEvent::from_fd(fd));
            event.writable = Some(true);
        }
        Ok(PollEvents {
            matched: &self.matched,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PollEvent {
    fd: RawFd,
    readable: Option<bool>,
    writable: Option<bool>,
}

impl PollEvent {
    pub fn new(fd: impl AsFd) -> Self {
        Self {
            fd: fd.as_fd().as_raw_fd(),
            readable: None,
            writable: None,
        }
    }

    fn from_fd(fd: RawFd) -> Self {
        Self {
            fd,
            readable: None,
            writable: None,
        }
    }

    pub fn with_readable(self, read: bool) -> Self {
        Self {
            readable: Some(read),
            ..self
        }
    }

    pub fn is_readable(&self) -> bool {
        self.readable.unwrap_or(false)
    }

    pub fn with_writable(self, write: bool) -> Self {
        Self {
            writable: Some(write),
            ..self
        }
    }

    pub fn is_writable(&self) -> bool {
        self.writable.unwrap_or(false)
    }

    pub fn is_unset(&self) -> bool {
        !self.is_writable() && !self.is_readable()
    }
}
pub struct PollEvents<'a> {
    matched: &'a HashMap<RawFd, PollEvent>,
}

impl PollEvents<'_> {
    pub fn get(&self, fd: impl AsFd) -> PollEvent {
        let fd = fd.as_fd().as_raw_fd();
        self.matched
            .get(&fd)
            .cloned()
            .unwrap_or_else(|| PollEvent::from_fd(fd))
    }

    pub fn len(&self) -> usize {
        self.matched.len()
    }
}
