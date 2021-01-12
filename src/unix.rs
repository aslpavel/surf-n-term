/// Unix systems specific `Terminal` implementation.
use crate::common::IOQueue;
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
use std::os::unix::io::AsRawFd;
use std::{
    collections::VecDeque,
    fs::File,
    io::{BufWriter, Cursor, Read, Write},
    path::Path,
    time::{Duration, Instant},
};

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
        unistd::{isatty, read, write},
        Error,
    };
    pub use std::os::unix::{io::RawFd, net::UnixStream};
}

pub struct UnixTerminal {
    write_handle: IOHandle,
    encoder: TTYEncoder,
    write_queue: IOQueue,
    read_handle: IOHandle,
    decoder: TTYDecoder,
    events_queue: VecDeque<TerminalEvent>,
    waker_read: nix::UnixStream,
    waker: TerminalWaker,
    termios_saved: nix::Termios,
    signal_delivery: SignalDelivery<nix::UnixStream, SignalOnly>,
    stats: TerminalStats,
    tee: Option<BufWriter<File>>,
    image_handler: Box<dyn ImageHandler + 'static>,
}

impl UnixTerminal {
    /// Create new terminal by opening `/dev/tty` device.
    pub fn new() -> Result<Self, Error> {
        Self::open("/dev/tty")
    }

    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let fd = nix::open(path.as_ref(), nix::OFlag::O_RDWR, nix::Mode::empty())?;
        Self::new_from_fd(fd, fd)
    }

    /// Create new terminal from raw file descriptors.
    pub fn new_from_fd(write_fd: nix::RawFd, read_fd: nix::RawFd) -> Result<Self, Error> {
        if !nix::isatty(write_fd)? || !nix::isatty(read_fd)? {
            return Err(Error::NotATTY);
        }

        // switching terminal into a raw mode
        // [Entering Raw Mode](https://viewsourcecode.org/snaptoken/kilo/02.enteringRawMode.html)
        let termios_saved = nix::tcgetattr(write_fd)?;
        let mut termios = termios_saved.clone();
        nix::cfmakeraw(&mut termios);
        nix::tcsetattr(write_fd, nix::SetArg::TCSAFLUSH, &termios)?;

        // signal delivery
        let (signal_read, signal_write) = nix::UnixStream::pair()?;
        let signal_delivery = SignalDelivery::with_pipe(
            signal_read,
            signal_write,
            SignalOnly,
            &[SIGWINCH, SIGTERM, SIGINT, SIGQUIT],
        )?;

        // self-pipe trick to implement waker
        let (waker_read, waker_write) = nix::UnixStream::pair()?;
        set_blocking(waker_write.as_raw_fd(), false)?;
        let waker = TerminalWaker::new(move || {
            const WAKE: &[u8] = b"\x00";
            // use write syscall instead of locking so it would be safe to use in a signal handler
            match guard_nix(nix::write(waker_write.as_raw_fd(), WAKE), 0) {
                Ok(_) => Ok(()),
                Err(error) => Err(error.into()),
            }
        });

        let write_handle = IOHandle::new(write_fd);
        let read_handle = IOHandle::new(read_fd);

        read_handle.set_blocking(false)?;
        write_handle.set_blocking(false)?;
        set_blocking(waker_read.as_raw_fd(), false)?;

        let mut term = Self {
            write_handle,
            encoder: TTYEncoder::default(),
            write_queue: Default::default(),
            read_handle,
            decoder: TTYDecoder::new(),
            events_queue: Default::default(),
            waker_read,
            waker,
            termios_saved,
            signal_delivery,
            stats: TerminalStats::new(),
            tee: None,
            image_handler: Box::new(DummyImageHandler),
        };
        term.image_handler = image_handler_detect(&mut term)?;
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

    pub fn image_handler(&mut self) -> &mut dyn ImageHandler {
        &mut self.image_handler
    }

    fn dispose(&mut self) -> Result<(), Error> {
        self.frames_drop();

        // revert descriptors to blocking mode
        self.write_handle.set_blocking(true)?;
        self.read_handle.set_blocking(true)?;

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

        // disable signal handler
        self.signal_delivery.handle().close();

        // restore termios settings
        nix::tcsetattr(
            self.write_handle.as_raw_fd(),
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
        let write_fd = self.write_handle.as_raw_fd();
        let read_fd = self.read_handle.as_raw_fd();
        let signal_fd = self.signal_delivery.get_read().as_raw_fd();
        let waker_fd = self.waker_read.as_raw_fd();

        let timeout_instant = timeout.map(|dur| Instant::now() + dur);
        let mut first_loop = true; // execute first loop even if timeout is 0
        while !self.write_queue.is_empty() || self.events_queue.is_empty() {
            // update descriptors sets
            read_set.clear();
            read_set.insert(read_fd);
            read_set.insert(signal_fd);
            read_set.insert(waker_fd);
            write_set.clear();
            if !self.write_queue.is_empty() {
                write_set.insert(write_fd);
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
            let _count = guard_nix(select, 0)?;

            // process pending output
            if write_set.contains(write_fd) {
                let write_handle = &mut self.write_handle;
                let tee = self.tee.as_mut();
                let send = self.write_queue.consume_with(|slice| {
                    let size = guard_io(write_handle.write(slice), 0)?;
                    tee.map(|tee| tee.write(&slice[..size])).transpose()?;
                    Ok::<_, Error>(size)
                })?;
                self.stats.send += send;
            }
            // process SIGWINCH
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
            if read_set.contains(read_fd) {
                let mut buf = [0u8; 1024];
                let recv = guard_io(self.read_handle.read(&mut buf), 0)?;
                self.stats.recv += recv;
                // parse events
                let mut read_queue = Cursor::new(&buf[..recv]);
                while let Some(event) = self.decoder.decode(&mut read_queue)? {
                    if !self.image_handler.handle(&event)? {
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
        match cmd {
            TerminalCommand::Image(image) => self.image_handler.draw(&image, &mut self.write_queue),
            TerminalCommand::ImageErase(pos) => {
                self.image_handler.erase(pos, &mut self.write_queue)
            }
            cmd => self.encoder.encode(&mut self.write_queue, cmd),
        }
    }

    fn size(&self) -> Result<TerminalSize, Error> {
        unsafe {
            let mut winsize: nix::winsize = std::mem::zeroed();
            if libc::ioctl(self.write_handle.as_raw_fd(), nix::TIOCGWINSZ, &mut winsize) < 0 {
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
}

/// Guard against EAGAIN and EINTR
fn guard_nix<T>(result: Result<T, nix::Error>, value: T) -> Result<T, nix::Error> {
    match result {
        Err(nix::Error::Sys(nix::Errno::EINTR)) | Err(nix::Error::Sys(nix::Errno::EAGAIN)) => {
            Ok(value)
        }
        _ => result,
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
fn set_blocking(fd: nix::RawFd, blocking: bool) -> Result<(), nix::Error> {
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
    fd: nix::RawFd,
}

impl IOHandle {
    pub fn new(fd: nix::RawFd) -> Self {
        Self { fd }
    }

    pub fn set_blocking(&self, blocking: bool) -> Result<(), nix::Error> {
        set_blocking(self.fd, blocking)
    }
}

impl AsRawFd for IOHandle {
    fn as_raw_fd(&self) -> nix::RawFd {
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
