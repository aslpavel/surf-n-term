//! Unix systems specific `Terminal` implementation.
use crate::common::IOQueue;
use crate::encoder::ColorDepth;
use crate::image::ImageHandlerKind;
use crate::TerminalCaps;
use crate::{
    decoder::{Decoder, TTYDecoder},
    encoder::{Encoder, TTYEncoder},
    error::Error,
    image::{image_handler_detect, DummyImageHandler},
    terminal::{
        Size, Terminal, TerminalCommand, TerminalEvent, TerminalSize, TerminalStats, TerminalWaker,
    },
    DecMode, ImageHandler,
};
use signal_hook::{
    consts::{SIGINT, SIGQUIT, SIGTERM, SIGWINCH},
    iterator::{backend::SignalDelivery, exfiltrator::SignalOnly},
};
use std::os::unix::{
    io::{AsRawFd, RawFd},
    net::UnixStream,
};
use std::{
    collections::VecDeque,
    fs::File,
    io::{BufWriter, Cursor, Read, Write},
    path::Path,
    time::{Duration, Instant},
};
use tracing::info;

mod nix {
    pub use libc::{winsize, TIOCGWINSZ};
    pub use nix::{
        errno::Errno,
        fcntl::{fcntl, open, FcntlArg, OFlag},
        sys::{
            select::{select, FdSet},
            stat::Mode,
            termios::{cfmakeraw, tcgetattr, tcsetattr, SetArg, Termios},
            time::TimeVal,
        },
        unistd::{close, isatty, read, write},
        Error,
    };
}

pub struct UnixTerminal {
    tty_handle: IOHandle,
    encoder: TTYEncoder,
    write_queue: IOQueue,
    decoder: TTYDecoder,
    events_queue: VecDeque<TerminalEvent>,
    waker_read: UnixStream,
    waker: TerminalWaker,
    termios_saved: nix::Termios,
    signal_delivery: SignalDelivery<UnixStream, SignalOnly>,
    stats: TerminalStats,
    tee: Option<BufWriter<File>>,
    image_handler: Box<dyn ImageHandler + 'static>,
    capabilities: TerminalCaps,
}

impl UnixTerminal {
    /// Create new terminal by opening `/dev/tty` device.
    pub fn new() -> Result<Self, Error> {
        Self::open("/dev/tty")
    }

    /// Open terminal by a given device path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let tty_fd = nix::open(path.as_ref(), nix::OFlag::O_RDWR, nix::Mode::empty())?;
        Self::new_from_fd(tty_fd)
    }

    /// Create new terminal from raw file descriptor pointing to /dev/tty.
    pub fn new_from_fd(tty_fd: RawFd) -> Result<Self, Error> {
        let tty_handle = IOHandle::new(tty_fd);
        tty_handle.set_blocking(false)?;
        if !nix::isatty(tty_fd)? {
            return Err(Error::NotATTY);
        }

        // switching terminal into a raw mode
        // [Entering Raw Mode](https://viewsourcecode.org/snaptoken/kilo/02.enteringRawMode.html)
        let termios_saved = nix::tcgetattr(tty_fd)?;
        let mut termios = termios_saved.clone();
        nix::cfmakeraw(&mut termios);
        nix::tcsetattr(tty_fd, nix::SetArg::TCSAFLUSH, &termios)?;

        // signal delivery
        let (signal_read, signal_write) = UnixStream::pair()?;
        let signal_delivery = SignalDelivery::with_pipe(
            signal_read,
            signal_write,
            SignalOnly,
            &[SIGWINCH, SIGTERM, SIGINT, SIGQUIT],
        )?;

        // self-pipe trick to implement waker
        let (waker_read, waker_write) = UnixStream::pair()?;
        set_blocking(waker_write.as_raw_fd(), false)?;
        let waker = TerminalWaker::new(move || {
            const WAKE: &[u8] = b"\x00";
            // use write syscall instead of locking so it would be safe to use in a signal handler
            match nix::write(waker_write.as_raw_fd(), WAKE) {
                Ok(_) | Err(nix::Error::Sys(nix::Errno::EINTR | nix::Errno::EAGAIN)) => Ok(()),
                Err(error) => Err(error.into()),
            }
        });
        set_blocking(waker_read.as_raw_fd(), false)?;

        let capabilities = TerminalCaps::default();
        Self {
            tty_handle,
            encoder: TTYEncoder::new(capabilities.clone()),
            write_queue: Default::default(),
            decoder: TTYDecoder::new(),
            events_queue: Default::default(),
            waker_read,
            waker,
            termios_saved,
            signal_delivery,
            stats: TerminalStats::new(),
            tee: None,
            image_handler: Box::new(DummyImageHandler),
            capabilities,
        }
        .detect()
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

    /// Detect termincal capabilieties
    fn detect(mut self) -> Result<Self, Error> {
        // color depth
        let depth = match std::env::var("COLORTERM") {
            Ok(value) if value == "truecolor" || value == "24bit" => ColorDepth::TrueColor,
            _ => ColorDepth::EightBit,
        };
        let depth = match std::env::var("TERM").as_deref() {
            Ok("linux") => ColorDepth::Gray,
            _ => depth,
        };

        // image handler
        let image_handler = image_handler_detect(&mut self)?;

        // update capabilieties
        self.capabilities.depth = depth;
        self.capabilities.glyphs = matches!(
            image_handler.kind(),
            ImageHandlerKind::Kitty | ImageHandlerKind::Sixel
        );

        // update fields
        self.encoder = TTYEncoder::new(self.capabilities.clone());
        self.image_handler = image_handler;

        info!("capabilities: {:?}", self.capabilities);
        Ok(self)
    }

    /// Close all descriptors free all the resources
    fn dispose(&mut self) -> Result<(), Error> {
        self.frames_drop();

        // revert descriptor to blocking mode
        self.tty_handle.set_blocking(true)?;

        // flush currently queued output and submit the epilogue
        let epilogue = [
            TerminalCommand::Face(Default::default()),
            TerminalCommand::DecModeSet {
                enable: true,
                mode: DecMode::VisibleCursor,
            },
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
        ];
        epilogue
            .iter()
            .try_fold((), |_, cmd| self.execute(cmd.clone()))
            .and_then(|_| {
                while !self.write_queue.is_empty() {
                    self.poll(Some(Duration::new(0, 0)))?;
                }
                Ok(())
            })
            .unwrap_or(()); // ignore write errors
        self.drain().count(); // drain pending events

        // disable signal handler
        self.signal_delivery.handle().close();

        // restore termios settings
        nix::tcsetattr(
            self.tty_handle.as_raw_fd(),
            nix::SetArg::TCSAFLUSH,
            &self.termios_saved,
        )?;

        Ok(())
    }
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
    fn poll(&mut self, timeout: Option<Duration>) -> Result<Option<TerminalEvent>, Error> {
        // NOTE:
        // Only `select` reliably works with /dev/tty on MacOS, `poll` for example
        // always returns POLLNVAL.
        self.write_queue.flush()?;
        let mut read_set = nix::FdSet::new();
        let mut write_set = nix::FdSet::new();
        let tty_fd = self.tty_handle.as_raw_fd();
        let signal_fd = self.signal_delivery.get_read().as_raw_fd();
        let waker_fd = self.waker_read.as_raw_fd();

        let timeout_instant = timeout.map(|dur| Instant::now() + dur);
        let mut first_loop = true; // execute first loop even if timeout is 0
        while !self.write_queue.is_empty() || self.events_queue.is_empty() {
            // update descriptors sets
            read_set.clear();
            read_set.insert(tty_fd);
            read_set.insert(signal_fd);
            read_set.insert(waker_fd);
            write_set.clear();
            if !self.write_queue.is_empty() {
                write_set.insert(tty_fd);
            }

            // process timeout
            let mut delay = match timeout_instant {
                Some(timeout_instant) => {
                    let now = Instant::now();
                    if timeout_instant < Instant::now() {
                        if first_loop {
                            Some(timeval_from_duration(Duration::new(0, 0)))
                        } else {
                            break;
                        }
                    } else {
                        Some(timeval_from_duration(timeout_instant - now))
                    }
                }
                None => None,
            };

            // wait for descriptors
            let select = nix::select(None, &mut read_set, &mut write_set, None, &mut delay);
            match select {
                Err(nix::Error::Sys(nix::Errno::EINTR | nix::Errno::EAGAIN)) => return Ok(None),
                Err(error) => return Err(error.into()),
                Ok(count) => tracing::trace!("select count={}", count),
            };

            // process pending output
            if write_set.contains(tty_fd) {
                let tee = self.tee.as_mut();
                let send = self.write_queue.consume_with(|slice| {
                    let size = guard_io(self.tty_handle.write(slice), 0)?;
                    tee.map(|tee| tee.write(&slice[..size])).transpose()?;
                    Ok::<_, Error>(size)
                })?;
                self.stats.send += send;
            }
            // process signals
            if read_set.contains(signal_fd) {
                for signal in self.signal_delivery.pending() {
                    match signal {
                        SIGWINCH => {
                            self.events_queue
                                .push_back(TerminalEvent::Resize(self.size()?));
                        }
                        SIGTERM | SIGINT | SIGQUIT => {
                            return Err(Error::Quit);
                        }
                        _ => {}
                    }
                }
            }
            // process waker
            if read_set.contains(waker_fd) {
                let mut buf = [0u8; 1024];
                if guard_io(self.waker_read.read(&mut buf), 0)? != 0 {
                    self.events_queue.push_back(TerminalEvent::Wake);
                }
            }
            // process pending input
            if read_set.contains(tty_fd) {
                let mut buf = [0u8; 1024];
                let recv = guard_io(self.tty_handle.read(&mut buf), 0)?;
                if recv == 0 {
                    return Err(Error::Quit);
                }
                self.stats.recv += recv;
                // parse events
                let mut read_queue = Cursor::new(&buf[..recv]);
                while let Some(event) = self.decoder.decode(&mut read_queue)? {
                    if !self.image_handler.handle(&event)? {
                        self.events_queue.push_back(event)
                    }
                }
                // Durty hack to exctract ambiguous terminal events (such as Escape key)
                // we assume that ambiguous events are never split across reads.
                if let Some(event) = self.decoder.take() {
                    self.events_queue.push_back(event);
                }
            }

            // indicate that first loop was executed
            first_loop = false;
        }
        Ok(self.events_queue.pop_front())
    }

    fn execute(&mut self, cmd: TerminalCommand) -> Result<(), Error> {
        tracing::trace!(?cmd, "execute");
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
        unsafe {
            let mut winsize: nix::winsize = std::mem::zeroed();
            if libc::ioctl(self.tty_handle.as_raw_fd(), nix::TIOCGWINSZ, &mut winsize) < 0 {
                return Err(nix::Error::last().into());
            }
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

/// Enable/disable blocking io for the provided file descriptor.
fn set_blocking(fd: RawFd, blocking: bool) -> Result<(), nix::Error> {
    let mut flags = nix::OFlag::from_bits_truncate(nix::fcntl(fd, nix::FcntlArg::F_GETFL)?);
    flags.set(nix::OFlag::O_NONBLOCK, !blocking);
    nix::fcntl(fd, nix::FcntlArg::F_SETFL(flags))?;
    Ok(())
}

fn timeval_from_duration(dur: Duration) -> nix::TimeVal {
    nix::TimeVal::from(libc::timeval {
        tv_sec: dur.as_secs() as libc::time_t,
        tv_usec: dur.subsec_micros() as libc::suseconds_t,
    })
}

struct IOHandle {
    fd: RawFd,
}

impl IOHandle {
    pub fn new(fd: RawFd) -> Self {
        Self { fd }
    }

    pub fn set_blocking(&self, blocking: bool) -> Result<(), nix::Error> {
        set_blocking(self.fd, blocking)
    }
}

impl Drop for IOHandle {
    fn drop(&mut self) {
        let _ = nix::close(self.fd);
    }
}

impl AsRawFd for IOHandle {
    fn as_raw_fd(&self) -> RawFd {
        self.fd
    }
}

impl Write for IOHandle {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        nix::write(self.fd, buf).map_err(|_| std::io::Error::last_os_error())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Read for IOHandle {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        nix::read(self.fd, buf).map_err(|_| std::io::Error::last_os_error())
    }
}
