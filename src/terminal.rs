use crate::{Face, Surface, View};
use std::{
    collections::VecDeque,
    fmt,
    io::{BufRead, Read, Write},
    time::Duration,
};

mod nix {
    pub use libc::{winsize, TIOCGWINSZ};
    pub use nix::{
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
    pub use std::os::unix::io::RawFd;
}

/*
enum TerminalCommand {}
enum TerminalEvent {}

trait Terminal: Sync + Write {
    fn write(&mut self, command: TerminalCommand) -> Result<(), TerminalError>;
    fn flush(&mut self) -> Result<(), TerminalError>;

    fn recv<F, FR>(&mut self, f: F) -> FR
    where
        F: FnMut(&mut Vec<TerminalEvent>) -> FR;
}

type BoxTerminal = std::boxed::Box<dyn Terminal>;
*/

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TerminalSize {
    pub width: usize,
    pub height: usize,
    pub width_pixels: usize,
    pub height_pixels: usize,
}

pub struct UnixTerminal {
    write_fd: nix::RawFd,
    write_queue: IOQueue,
    read_fd: nix::RawFd,
    read_queue: IOQueue,
    termios_saved: nix::Termios,
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

        set_blocking(read_fd, false)?;
        set_blocking(write_fd, false)?;

        Ok(Self {
            write_fd,
            write_queue: Default::default(),
            read_fd,
            read_queue: Default::default(),
            termios_saved,
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

    pub fn wait(&mut self, timeout: Option<Duration>) -> Result<(), TerminalError> {
        // NOTE:
        // Only `select` reliably works with /dev/tty on MacOS, poll for example
        // always returns POLLNVAL.
        self.write_queue.flush()?;
        let mut read_set = nix::FdSet::new();
        let mut write_set = nix::FdSet::new();
        let write_fd = self.write_fd;

        while !self.write_queue.is_empty() || self.read_queue.is_empty() {
            read_set.clear();
            write_set.clear();
            read_set.insert(self.read_fd);
            if !self.write_queue.is_empty() {
                write_set.insert(self.write_fd);
            }
            let mut timeout = timeout.map(timeval_from_duration);
            let _count = nix::select(None, &mut read_set, &mut write_set, None, &mut timeout)?;
            if !self.write_queue.is_empty() && write_set.contains(self.write_fd) {
                self.write_queue
                    .consume_with(|slice| nix::write(write_fd, slice))?;
            }
            if read_set.contains(self.read_fd) {
                let mut buf = [0u8; 1024];
                let size = nix::read(self.read_fd, &mut buf)?;
                self.read_queue.write_all(&buf[..size])?;
            }
            println!("\x1b[31;01mloop\x1b[m");
        }
        Ok(())
    }

    pub fn write(&mut self, data: impl AsRef<[u8]>) -> Result<(), TerminalError> {
        self.write_queue.write_all(data.as_ref())?;
        Ok(())
    }

    pub fn debug(&mut self) -> Result<(), TerminalError> {
        let mut read = String::new();
        self.read_queue.read_to_string(&mut read)?;
        println!("\x1b[32;01mREAD\x1b[m {:?}", read);
        Ok(())
    }
}

impl std::ops::Drop for UnixTerminal {
    fn drop(&mut self) {
        set_blocking(self.write_fd, false).unwrap_or(());
        set_blocking(self.read_fd, false).unwrap_or(());
        // restore termios settings
        nix::tcsetattr(self.write_fd, nix::SetArg::TCSAFLUSH, &self.termios_saved).unwrap_or(());
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

struct IOQueue {
    chunks: VecDeque<Vec<u8>>,
    offset: usize,
}

impl IOQueue {
    fn new() -> Self {
        Self {
            chunks: Default::default(),
            offset: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Remaining slice at the front of the queue
    pub fn as_slice(&self) -> &[u8] {
        match self.chunks.front() {
            Some(chunk) => &chunk.as_slice()[self.offset..],
            None => &[],
        }
    }

    /// Consume bytes from the front of the queue
    pub fn consume(&mut self, amt: usize) {
        if self.chunks.front().map(|chunk| chunk.len()).unwrap_or(0) > self.offset + amt {
            self.offset += amt
        } else {
            self.offset = 0;
            self.chunks.pop_front();
        }
    }

    pub fn consume_with<F, FE>(&mut self, consumer: F) -> Result<usize, FE>
    where
        F: FnOnce(&[u8]) -> Result<usize, FE>,
    {
        let size = consumer(self.as_slice())?;
        self.consume(size);
        Ok(size)
    }
}

impl Write for IOQueue {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.chunks.is_empty() {
            self.flush()?;
        }
        let chunk = self.chunks.back_mut().unwrap();
        chunk.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.chunks.push_back(Default::default());
        Ok(())
    }
}

impl Read for IOQueue {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let buf_queue = self.as_slice();
        let size = std::cmp::min(buf.len(), buf_queue.len());
        let src = &buf_queue[..size];
        let dst = &mut buf[..size];
        dst.copy_from_slice(src);
        self.consume(size);
        Ok(size)
    }
}

impl BufRead for IOQueue {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        Ok(self.as_slice())
    }

    fn consume(&mut self, amt: usize) {
        Self::consume(self, amt)
    }
}

impl Default for IOQueue {
    fn default() -> Self {
        IOQueue::new()
    }
}

#[cfg(test)]
mod tests {
    use super::IOQueue;
    use std::io::{BufRead, Read, Write};

    #[test]
    fn test_io_queue() -> std::io::Result<()> {
        let mut queue = IOQueue::new();
        assert!(queue.is_empty());

        queue.write(b"on")?;
        queue.write(b"e")?;
        queue.flush()?;
        queue.write(b",two")?;
        queue.flush()?;
        assert!(!queue.is_empty());

        // make sure flush creates separate chunks
        assert_eq!(queue.as_slice(), b"one");

        // check `Read` implementation
        let mut out = String::new();
        queue.read_to_string(&mut out)?;
        assert_eq!(out, String::from("one,two"));
        assert!(queue.is_empty());

        // make `BufRead` implementation
        queue.write(b"one\nt")?;
        queue.flush()?;
        queue.write(b"wo\nth")?;
        queue.flush()?;
        queue.write(b"ree\n")?;
        let lines = queue.lines().collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            lines,
            vec!["one".to_string(), "two".to_string(), "three".to_string()]
        );
        Ok(())
    }
}

// ----------------------------------------------------------------------------------------------------
pub trait Renderer {
    type Error;
    fn render(&mut self, surface: &Surface) -> Result<(), Self::Error>;
}

#[derive(Debug)]
pub enum TerminalError {
    IOError(std::io::Error),
    NixError(nix::Error),
    Closed,
    NotATTY,
}

impl fmt::Display for TerminalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for TerminalError {}

impl From<std::io::Error> for TerminalError {
    fn from(error: std::io::Error) -> Self {
        Self::IOError(error)
    }
}

impl From<nix::Error> for TerminalError {
    fn from(error: nix::Error) -> Self {
        Self::NixError(error)
    }
}

pub struct TerminalRenderer {
    face: Face,
    queue: Vec<u8>,
    tty: std::io::Stdout, // FIXME: use `/dev/tty` instead
}

impl TerminalRenderer {
    pub fn new() -> Result<Self, TerminalError> {
        Ok(Self {
            face: Default::default(),
            queue: Vec::new(),
            tty: std::io::stdout(),
        })
    }

    // FIXME: support for different color depth and attributes
    fn set_face(&mut self, face: Face) -> Result<(), TerminalError> {
        if self.face == face {
            return Ok(());
        }
        write!(&mut self.queue, "\x1b[00")?;
        if self.face.bg != face.bg {
            face.bg
                .map(|c| write!(self.queue, ";48;2;{};{};{}", c.red, c.green, c.blue))
                .transpose()?;
        }
        if self.face.fg != face.fg {
            face.fg
                .map(|c| write!(self.queue, ";38;2;{};{};{}", c.red, c.green, c.blue))
                .transpose()?;
        }
        if self.face.attrs != face.attrs {
            unimplemented!()
        }
        write!(self.queue, "m")?;
        self.face = face;
        Ok(())
    }

    fn set_cursor(&mut self, row: usize, col: usize) -> Result<(), TerminalError> {
        write!(self.queue, "\x1b[{};{}H", row + 1, col + 1)?;
        Ok(())
    }
}

impl Renderer for TerminalRenderer {
    type Error = TerminalError;

    fn render(&mut self, surface: &Surface) -> Result<(), Self::Error> {
        let shape = surface.shape();
        let data = surface.data();
        for row in 0..shape.height {
            self.set_cursor(row, 0)?;
            for col in 0..shape.width {
                let cell = &data[shape.index(row, col)];
                self.set_face(cell.face)?;
                match cell.glyph {
                    Some(glyph) => self.queue.write(&glyph)?,
                    None => self.queue.write(&[b' '])?,
                };
            }
        }
        self.set_face(Face::default())?;

        self.tty.write_all(&self.queue)?;
        self.tty.flush()?;
        self.queue.clear();

        Ok(())
    }
}
