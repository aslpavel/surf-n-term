use super::{
    common::IOQueue, Decoder, Renderer, TTYDecoder, Terminal, TerminalCommand, TerminalError,
    TerminalEvent, TerminalSize,
};
use crate::{Face, FaceAttrs, Surface, View};
use std::os::unix::io::AsRawFd;
use std::{
    collections::VecDeque,
    io::{Cursor, Write},
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
    write_fd: nix::RawFd,
    write_queue: IOQueue,
    read_fd: nix::RawFd,
    decoder: TTYDecoder,
    events_queue: VecDeque<TerminalEvent>,
    termios_saved: nix::Termios,
    sigwinch_read: nix::UnixStream,
    sigwinch_id: signal_hook::SigId,
}

impl UnixTerminal {
    pub fn new_from_fd(write_fd: nix::RawFd, read_fd: nix::RawFd) -> Result<Self, TerminalError> {
        if !nix::isatty(write_fd)? || !nix::isatty(read_fd)? {
            return Err(TerminalError::NotATTY);
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

        set_blocking(read_fd, false)?;
        set_blocking(write_fd, false)?;
        set_blocking(sigwinch_read.as_raw_fd(), false)?;

        Ok(Self {
            write_fd,
            write_queue: Default::default(),
            read_fd,
            decoder: TTYDecoder::new(),
            events_queue: Default::default(),
            termios_saved,
            sigwinch_read,
            sigwinch_id,
        })
    }

    pub fn new() -> Result<Self, TerminalError> {
        let tty = nix::open("/dev/tty", nix::OFlag::O_RDWR, nix::Mode::empty())?;
        Self::new_from_fd(tty, tty)
    }

    pub fn size(&self) -> Result<TerminalSize, TerminalError> {
        unsafe {
            let mut winsize: nix::winsize = std::mem::zeroed();
            if libc::ioctl(self.write_fd, nix::TIOCGWINSZ, &mut winsize) < 0 {
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

impl std::ops::Drop for UnixTerminal {
    fn drop(&mut self) {
        set_blocking(self.write_fd, true).unwrap_or(());
        set_blocking(self.read_fd, true).unwrap_or(());

        // disable SIGIWNCH handler
        signal_hook::unregister(self.sigwinch_id);

        // restore termios settings
        nix::tcsetattr(self.write_fd, nix::SetArg::TCSAFLUSH, &self.termios_saved).unwrap_or(());
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
    fn poll(&mut self, timeout: Option<Duration>) -> Result<Option<TerminalEvent>, TerminalError> {
        // NOTE:
        // Only `select` reliably works with /dev/tty on MacOS, `poll` for example
        // always returns POLLNVAL.
        self.write_queue.flush()?;
        let mut read_set = nix::FdSet::new();
        let mut write_set = nix::FdSet::new();
        let write_fd = self.write_fd;
        let sigwinch_fd = self.sigwinch_read.as_raw_fd();

        let timeout_instant = timeout.map(|dur| Instant::now() + dur);
        let mut first_loop = true; // execute first loop even if timeout is 0
        while !self.write_queue.is_empty() || self.events_queue.is_empty() {
            // update descriptors sets
            read_set.clear();
            read_set.insert(self.read_fd);
            read_set.insert(sigwinch_fd);
            write_set.clear();
            if !self.write_queue.is_empty() {
                write_set.insert(self.write_fd);
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
            let _count = guard_io(select, 0)?;

            // process pending output
            if write_set.contains(self.write_fd) {
                self.write_queue
                    .consume_with(|slice| guard_io(nix::write(write_fd, slice), 0))?;
            }
            // process SIGWINCH
            if read_set.contains(sigwinch_fd) {
                let mut buf = [0u8; 1];
                if guard_io(nix::read(sigwinch_fd, &mut buf), 0)? != 0 {
                    self.events_queue
                        .push_back(TerminalEvent::Resize(self.size()?));
                }
            }
            // process pending input
            if read_set.contains(self.read_fd) {
                let mut buf = [0u8; 1024];
                let size = guard_io(nix::read(self.read_fd, &mut buf), 0)?;
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

    fn execute(&mut self, cmd: TerminalCommand) -> Result<(), TerminalError> {
        use TerminalCommand::*;

        match cmd {
            DecModeSet { enable, mode } => {
                let flag = if enable { "h" } else { "l" };
                write!(self, "\x1b[?{}{}", mode as usize, flag)?;
            }
            DecModeReport(mode) => {
                write!(self, "\x1b[?{}$p", mode as usize)?;
            }
            CursorTo { row, col } => write!(self, "\x1b[{};{}H", row, col)?,
            CursorReport => self.write_all(b"\x1b[6n")?,
            CursorSave => self.write_all(b"\x1b[s")?,
            CursorRestore => self.write_all(b"\x1b[u")?,
            EraseLineRight => self.write_all(b"\x1b[K")?,
            EraseLineLeft => self.write_all(b"\x1b[1K")?,
            EraseLine => self.write_all(b"\x1b[2K")?,
            Face(face) => {
                self.write_all(b"\x1b[00")?;
                if let Some(fg) = face.fg {
                    let (r, g, b) = fg.rgb_u8();
                    write!(self, ";38;2;{};{};{}", r, g, b)?;
                }
                if let Some(bg) = face.bg {
                    let (r, g, b) = bg.rgb_u8();
                    write!(self, ";48;2;{};{};{}", r, g, b)?;
                }
                if !face.attrs.is_empty() {
                    for (flag, code) in &[
                        (FaceAttrs::BOLD, b";01"),
                        (FaceAttrs::ITALIC, b";03"),
                        (FaceAttrs::UNDERLINE, b";04"),
                        (FaceAttrs::BLINK, b";05"),
                        (FaceAttrs::REVERSE, b";07"),
                    ] {
                        if face.attrs.contains(*flag) {
                            self.write_all(*code)?;
                        }
                    }
                }
                self.write_all(b"m")?;
            }
            Reset => self.write_all(b"\x1bc")?,
        }

        Ok(())
    }
}

impl Renderer for UnixTerminal {
    fn render(&mut self, surface: &Surface) -> Result<(), TerminalError> {
        let mut cur_face = Face::default();
        let shape = surface.shape();
        let data = surface.data();
        self.execute(TerminalCommand::CursorSave)?;
        for row in 0..shape.height {
            self.execute(TerminalCommand::CursorTo { row, col: 0 })?;
            for col in 0..shape.width {
                let cell = &data[shape.index(row, col)];
                if cur_face != cell.face {
                    self.execute(TerminalCommand::Face(cell.face))?;
                    cur_face = cell.face;
                }
                match cell.glyph {
                    Some(glyph) => self.write_all(&glyph)?,
                    None => self.write_all(&[b' '])?,
                };
            }
        }
        self.execute(TerminalCommand::CursorRestore)?;
        self.flush()?;
        Ok(())
    }
}

/// Guard against EAGAIN and EINTR
fn guard_io<T>(result: Result<T, nix::Error>, value: T) -> Result<T, nix::Error> {
    match result {
        Err(nix::Error::Sys(nix::Errno::EINTR)) | Err(nix::Error::Sys(nix::Errno::EAGAIN)) => {
            Ok(value)
        }
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
