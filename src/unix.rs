/// Unix systems specific `Terminal` implementation.
use crate::common::IOQueue;
use crate::{
    decoder::{Decoder, TTYDecoder},
    encoder::{Encoder, TTYEncoder},
    error::Error,
    terminal::{Terminal, TerminalCommand, TerminalEvent, TerminalSize},
    DecMode,
};
use std::os::unix::io::AsRawFd;
use std::{
    collections::VecDeque,
    io::{Cursor, Read, Write},
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
            termios::{tcgetattr, tcsetattr, InputFlags, LocalFlags, OutputFlags, SetArg, Termios},
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
    termios_saved: nix::Termios,
    sigwinch_read: nix::UnixStream,
    sigwinch_id: signal_hook::SigId,
}

impl UnixTerminal {
    pub fn new_from_fd(write_fd: nix::RawFd, read_fd: nix::RawFd) -> Result<Self, Error> {
        if !nix::isatty(write_fd)? || !nix::isatty(read_fd)? {
            return Err(Error::NotATTY);
        }

        // switching terminal into a raw mode
        // [Entering Raw Mode](https://viewsourcecode.org/snaptoken/kilo/02.enteringRawMode.html)
        let termios_saved = nix::tcgetattr(write_fd)?;
        let mut termios = termios_saved.clone();
        // local flags
        termios.local_flags.remove(nix::LocalFlags::ICANON); // turn off `canonical mode`
        termios.local_flags.remove(nix::LocalFlags::ECHO); // do not echo back typed input
        termios.local_flags.remove(nix::LocalFlags::ISIG); // do not send signals on `ctrl-{c|z}`
        termios.local_flags.remove(nix::LocalFlags::IEXTEN); // disable literal mode `ctrl-v`
        termios.input_flags.remove(nix::InputFlags::IXON); // disable control flow `ctrl-{s|q}`
        termios.input_flags.remove(nix::InputFlags::ICRNL); // correctly handle ctrl-m

        // termios.output_flags.remove(nix::OutputFlags::OPOST); // do not post process `\n` input `\r\n`
        nix::tcsetattr(write_fd, nix::SetArg::TCSAFLUSH, &termios)?;

        // self-pipe trick to handle SIGWINCH (window resize) signal
        let (sigwinch_read, sigwinch_write) = nix::UnixStream::pair()?;
        let sigwinch_id = signal_hook::pipe::register(signal_hook::SIGWINCH, sigwinch_write)?;

        let write_handle = IOHandle::new(write_fd);
        let read_handle = IOHandle::new(read_fd);

        read_handle.set_blocking(false)?;
        write_handle.set_blocking(false)?;
        set_blocking(sigwinch_read.as_raw_fd(), false)?;

        Ok(Self {
            write_handle,
            encoder: TTYEncoder::new(),
            write_queue: Default::default(),
            read_handle,
            decoder: TTYDecoder::new(),
            events_queue: Default::default(),
            termios_saved,
            sigwinch_read,
            sigwinch_id,
        })
    }

    pub fn new() -> Result<Self, Error> {
        let tty = nix::open("/dev/tty", nix::OFlag::O_RDWR, nix::Mode::empty())?;
        Self::new_from_fd(tty, tty)
    }
}

impl std::ops::Drop for UnixTerminal {
    fn drop(&mut self) {
        self.write_handle.set_blocking(true).unwrap_or(());
        self.read_handle.set_blocking(true).unwrap_or(());

        // restore settings
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
        ];
        let mut buffer = Vec::new();
        for cmd in epilogue.iter() {
            self.encoder
                .encode(&mut buffer, *cmd)
                .expect("in-memory write failed");
        }
        self.write_handle.write_all(&buffer).unwrap_or(());

        // disable SIGIWNCH handler
        signal_hook::unregister(self.sigwinch_id);

        // restore termios settings
        nix::tcsetattr(
            self.write_handle.as_raw_fd(),
            nix::SetArg::TCSAFLUSH,
            &self.termios_saved,
        )
        .unwrap_or(());
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
        let sigwinch_fd = self.sigwinch_read.as_raw_fd();

        let timeout_instant = timeout.map(|dur| Instant::now() + dur);
        let mut first_loop = true; // execute first loop even if timeout is 0
        while !self.write_queue.is_empty() || self.events_queue.is_empty() {
            // update descriptors sets
            read_set.clear();
            read_set.insert(read_fd);
            read_set.insert(sigwinch_fd);
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
                self.write_queue
                    .consume_with(|slice| guard_io(write_handle.write(slice), 0))?;
            }
            // process SIGWINCH
            if read_set.contains(sigwinch_fd) {
                let mut buf = [0u8; 1];
                if guard_io(self.sigwinch_read.read(&mut buf), 0)? != 0 {
                    self.events_queue
                        .push_back(TerminalEvent::Resize(self.size()?));
                }
            }
            // process pending input
            if read_set.contains(read_fd) {
                let mut buf = [0u8; 1024];
                let size = guard_io(self.read_handle.read(&mut buf), 0)?;

                // parse events
                let mut read_queue = Cursor::new(&buf[..size]);
                while let Some(event) = self.decoder.decode(&mut read_queue)? {
                    self.events_queue.push_back(event)
                }
            }

            // indicate that first loop was executed
            first_loop = false;
        }
        Ok(self.events_queue.pop_front())
    }

    fn execute(&mut self, cmd: TerminalCommand) -> Result<(), Error> {
        self.encoder.encode(&mut self.write_queue, cmd)
    }

    fn size(&self) -> Result<TerminalSize, Error> {
        unsafe {
            let mut winsize: nix::winsize = std::mem::zeroed();
            if libc::ioctl(self.write_handle.as_raw_fd(), nix::TIOCGWINSZ, &mut winsize) < 0 {
                return Err(nix::Error::last().into());
            }
            Ok(TerminalSize {
                height: winsize.ws_row as usize,
                width: winsize.ws_col as usize,
                height_pixels: winsize.ws_ypixel as usize,
                width_pixels: winsize.ws_xpixel as usize,
            })
        }
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
