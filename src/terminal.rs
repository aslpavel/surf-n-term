use crate::{
    error::Error,
    render::{TerminalRenderer, TerminalSurface},
    Face, Image, Key, KeyMod, KeyName, RGBA,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    io::Write,
    sync::Arc,
    time::Duration,
};

/// Main trait to interact with a Terminal
pub trait Terminal: Write {
    /// Schedue command for execution
    ///
    /// Command will be submitted on the next call to poll `Terminal::poll`
    fn execute(&mut self, cmd: TerminalCommand) -> Result<(), Error>;

    /// Waker object
    ///
    /// Waker object is a thread safe object that can be called to wake terminal
    /// with TerminalEvent::Wake event
    fn waker(&self) -> TerminalWaker;

    /// Poll the Terminal
    ///
    /// Only this function actually reads from or writes to the terminal.
    /// None duration blocks indefinitely until event received from the terminal.
    fn poll(&mut self, timeout: Option<Duration>) -> Result<Option<TerminalEvent>, Error>;

    /// Get terminal size
    fn size(&self) -> Result<TerminalSize, Error>;

    /// Run terminal with event handler
    fn run<H, R, E>(&mut self, mut timeout: Option<Duration>, mut handler: H) -> Result<R, E>
    where
        H: FnMut(&mut Self, Option<TerminalEvent>) -> Result<TerminalAction<R>, E>,
        E: From<Error>,
        Self: Sized,
    {
        loop {
            let event = self.poll(timeout)?;
            timeout = match handler(self, event)? {
                TerminalAction::Quit(result) => return Ok(result),
                TerminalAction::Wait => None,
                TerminalAction::Sleep(timeout) => Some(timeout),
            };
        }
    }

    /// Run terminal with render event handler
    ///
    /// Handler accepts mutable reference to the terminal, event that
    /// trigered the handler and terminal surface that should be used to
    /// render current frame (on each frame, it operates in immediate mode).
    /// Renderer will calculcate the difference between new terminal surface
    /// and terminal surface on the previous frame and will issue appropirate
    /// terminal commands to produce the desired result.
    fn run_render<H, R, E>(&mut self, mut handler: H) -> Result<R, E>
    where
        H: for<'a> FnMut(
            &'a mut Self,
            Option<TerminalEvent>,
            TerminalSurface<'a>,
        ) -> Result<TerminalAction<R>, E>,
        E: From<Error>,
        Self: Sized,
    {
        let mut renderer = TerminalRenderer::new(self, false)?;
        // run with render event handler
        self.run(Some(Duration::new(0, 0)), move |term, event| {
            if let Some(TerminalEvent::Resize(_)) = event {
                renderer = TerminalRenderer::new(term, true)?;
            }
            let result = handler(term, event, renderer.view())?;
            renderer.frame(term)?;
            Ok(result)
        })
    }
}

#[derive(Clone)]
pub struct TerminalWaker {
    wake: Arc<dyn Fn() -> Result<(), Error> + Sync + Send + 'static>,
}

impl TerminalWaker {
    pub fn new(wake: impl Fn() -> Result<(), Error> + Sync + Send + 'static) -> Self {
        Self {
            wake: Arc::new(wake),
        }
    }

    pub fn wake(&self) -> Result<(), Error> {
        (self.wake)()
    }
}

pub enum TerminalAction<R> {
    Quit(R),
    Wait,
    Sleep(Duration),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminalCommand {
    /// Put character
    Char(char),
    /// Set current face (foreground/background colors and text attributes)
    Face(Face),
    /// Control specified DEC mode (DECSET|DECRST)
    DecModeSet {
        enable: bool,
        mode: DecMode,
    },
    /// Report specified DEC mode (DECRQM)
    DecModeGet(DecMode),
    /// Request current cursor postion
    CursorGet,
    /// Move cursor to specified row and column
    CursorTo(Position),
    /// Save current cursor position
    CursorSave,
    /// Restore previously saved cursor position
    CursorRestore,
    /// Erase line using current background color to the left of the cursor
    EraseLineLeft,
    /// Erase line using current background color to the right of the cursor
    EraseLineRight,
    /// Erase line using current background color
    EraseLine,
    /// Erase specified ammount of characters to the right from current cursor position
    EraseChars(usize),
    /// Scroll, positive is up and negative is down
    Scroll(i32),
    /// Full reset of the terminal
    Reset,
    /// Draw image
    Image(Image),
    /// Erase image
    ImageErase(Position),
    /// Request Termcap/Terminfo XTGETTCAP
    Termcap(Vec<String>),
    /// Set or query terminal colors
    Color {
        name: TerminalColor,
        color: Option<RGBA>,
    },
    // Set terminal title
    Title(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TerminalColor {
    Background,
    Foreground,
    Palette(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DecMode {
    /// Visibility of the cursor
    VisibleCursor = 25,
    /// Wrapping of the text when it reaches end of the line
    AutoWrap = 7,
    /// Enable/Disable mouse reporting
    MouseReport = 1000,
    /// Report mouse motion events if `MouseReport` is enabled
    MouseMotions = 1003,
    /// Report mouse event in SGR format
    MouseSGR = 1006,
    /// Alternative screen mode
    AltScreen = 1049,
    /// Kitty keyboard mode https://sw.kovidgoyal.net/kitty/protocol-extensions.html
    KittyKeyboard = 2017,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Position {
    pub row: usize,
    pub col: usize,
}

impl Position {
    pub fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }
}

impl DecMode {
    pub fn from_usize(code: usize) -> Option<Self> {
        use DecMode::*;
        for mode in [
            VisibleCursor,
            AutoWrap,
            MouseReport,
            MouseMotions,
            MouseSGR,
            AltScreen,
            KittyKeyboard,
        ]
        .iter()
        {
            if code == *mode as usize {
                return Some(*mode);
            }
        }
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DecModeStatus {
    NotRecognized = 0,
    Enabled = 1,
    Disabled = 2,
    PermanentlyEnabled = 3,
    PermanentlyDisabled = 4,
}

impl DecModeStatus {
    pub fn from_usize(code: usize) -> Option<Self> {
        use DecModeStatus::*;
        for status in [
            NotRecognized,
            Enabled,
            Disabled,
            PermanentlyEnabled,
            PermanentlyDisabled,
        ]
        .iter()
        {
            if code == *status as usize {
                return Some(*status);
            }
        }
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TerminalEvent {
    // Key press event
    Key(Key),
    // Mouse event
    Mouse(Mouse),
    // Current cursor position
    CursorPosition {
        row: usize,
        col: usize,
    },
    // Terminal was resized
    Resize(TerminalSize),
    // Current terminal size
    Size(TerminalSize),
    // DEC mode status
    DecMode {
        mode: DecMode,
        status: DecModeStatus,
    },
    // Kitty image result
    KittyImage {
        id: u64,
        error: Option<String>,
    },
    // Terminal have been woken by waker object
    Wake,
    // Termcap/Terminfo repsponse to XTGETTCAP
    Termcap(BTreeMap<String, Option<String>>),
    // Terminal Attributes DA1 response
    DeviceAttrs(BTreeSet<usize>),
    // Unrecognized bytes (TODO: remove Vec and just use u8)
    Raw(Vec<u8>),
    Color {
        name: TerminalColor,
        color: RGBA,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Size {
    pub height: usize,
    pub width: usize,
}

impl Size {
    pub fn empty() -> Self {
        Self::new(0, 0)
    }

    pub fn new(height: usize, width: usize) -> Self {
        Self { height, width }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TerminalSize {
    pub cells: Size,
    pub pixels: Size,
}

impl TerminalSize {
    pub fn cell_size(&self) -> Size {
        Size {
            height: self.pixels.height / self.cells.height,
            width: self.pixels.width / self.cells.width,
        }
    }

    pub fn cells_in_pixels(&self, cells: Size) -> Size {
        let cell_size = self.cell_size();
        Size {
            height: cells.height * cell_size.height,
            width: cells.width * cell_size.width,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TerminalStats {
    pub send: usize,
    pub recv: usize,
}

impl Default for TerminalStats {
    fn default() -> Self {
        Self::new()
    }
}

impl TerminalStats {
    pub fn new() -> Self {
        Self { send: 0, recv: 0 }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Mouse {
    pub name: KeyName,
    pub mode: KeyMod,
    pub row: usize,
    pub col: usize,
}

impl fmt::Debug for Mouse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.mode.is_empty() {
            write!(f, "{:?} [{},{}]", self.name, self.row, self.col)?;
        } else {
            write!(
                f,
                "{:?}+{:?} [{},{}]",
                self.name, self.mode, self.row, self.col
            )?;
        }
        Ok(())
    }
}
