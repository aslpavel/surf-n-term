//! Main interface to interact with terminal
use crate::{
    Face, FaceModify, Image, Key, KeyMod, KeyName, RGBA, SurfaceMut,
    encoder::ColorDepth,
    error::Error,
    render::{TerminalRenderer, TerminalSurface, TerminalSurfaceExt},
};
use either::Either;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    io::Write,
    str::FromStr,
    sync::Arc,
    time::Duration,
};

/// How many frames needs to be pending before we start dropping them
const TERMINAL_FRAMES_DROP: usize = 32;

/// Main trait to interact with a Terminal
pub trait Terminal: Write + Send {
    /// Schedule command for execution
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

    /// Schedule multiple commands at once
    fn execute_many(&mut self, cmds: impl IntoIterator<Item = TerminalCommand>) -> Result<(), Error>
    where
        Self: Sized,
    {
        cmds.into_iter().try_for_each(|cmd| self.execute(cmd))
    }

    /// Iterator over pending events
    fn drain(&mut self) -> TerminalDrain<'_> {
        TerminalDrain(self.dyn_ref())
    }

    /// Create dynamic reference to the terminal object
    fn dyn_ref(&mut self) -> &mut dyn Terminal;

    /// Get terminal size
    fn size(&self) -> Result<TerminalSize, Error>;

    /// Get cursor position
    fn position(&mut self) -> Result<Position, Error>;

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
                TerminalAction::Wait | TerminalAction::WaitNoFrame => None,
                TerminalAction::Sleep(timeout) => Some(timeout),
            };
        }
    }

    /// Run terminal with render event handler
    ///
    /// Handler accepts mutable reference to the terminal, event that
    /// triggered the handler and terminal surface that should be used to
    /// render current frame (on each frame, it operates in immediate mode).
    /// Renderer will calculate the difference between new terminal surface
    /// and terminal surface on the previous frame and will issue appropriate
    /// terminal commands to produce the desired result.
    fn run_render<H, R, E>(&mut self, mut handler: H) -> Result<R, E>
    where
        H: FnMut(
            &mut Self,
            Option<TerminalEvent>,
            TerminalSurface<'_>,
        ) -> Result<TerminalAction<R>, E>,
        E: From<Error>,
        Self: Sized,
    {
        let mut renderer = TerminalRenderer::new(self, false)?;
        let mut timeout = Some(Duration::new(0, 0)); // run first loop immediately
        loop {
            let event = self.poll(timeout);
            tracing::trace!("[Terminal.run_render] {:?}", event);
            match event {
                Err(error) => {
                    // cleanup on error
                    renderer.surface().erase(Face::default());
                    renderer.frame(self)?;
                    let _ = self.poll(Some(Duration::new(0, 0)));
                    return Err(error.into());
                }
                Ok(event) => {
                    // allocate new renderer on resize
                    if let Some(TerminalEvent::Resize(_)) = event {
                        renderer.clear(self)?;
                        renderer = TerminalRenderer::new(self, true)?;
                    }
                    // handle event
                    let action = handler(self, event, renderer.surface())?;
                    if !matches!(action, TerminalAction::WaitNoFrame) {
                        // drop frames if we are too far behind
                        if self.frames_pending() > TERMINAL_FRAMES_DROP {
                            tracing::warn!(
                                "[Terminal.run_render] dropping frames: {}",
                                self.frames_pending()
                            );
                            self.frames_drop();
                            renderer.clear(self)?;
                        }
                        // render frame
                        self.execute(TerminalCommand::DecModeSet {
                            enable: true,
                            mode: DecMode::SynchronizedOutput,
                        })?;
                        renderer.frame(self)?;
                        self.execute(TerminalCommand::DecModeSet {
                            enable: false,
                            mode: DecMode::SynchronizedOutput,
                        })?;
                    } else {
                        renderer.surface().clear();
                    }
                    // handle action
                    timeout = match action {
                        TerminalAction::Quit(result) => return Ok(result),
                        TerminalAction::Wait | TerminalAction::WaitNoFrame => None,
                        TerminalAction::Sleep(timeout) => Some(timeout),
                    };
                }
            }
        }
    }

    /// Number of pending frames (equal to number of flush calls) to be rendered
    ///
    /// This information can be useful to provide back pressure, if terminal
    /// is not fast enough.
    fn frames_pending(&self) -> usize;

    /// Drop all pending frames (equal to number of flush calls)
    fn frames_drop(&mut self);

    /// Get terminal capabilities
    fn capabilities(&self) -> &TerminalCaps;
}

impl<T: Terminal + ?Sized> Terminal for &mut T {
    fn execute(&mut self, cmd: TerminalCommand) -> Result<(), Error> {
        (**self).execute(cmd)
    }

    fn waker(&self) -> TerminalWaker {
        (**self).waker()
    }

    fn poll(&mut self, timeout: Option<Duration>) -> Result<Option<TerminalEvent>, Error> {
        (**self).poll(timeout)
    }

    fn dyn_ref(&mut self) -> &mut dyn Terminal {
        (**self).dyn_ref()
    }

    fn size(&self) -> Result<TerminalSize, Error> {
        (**self).size()
    }

    fn position(&mut self) -> Result<Position, Error> {
        (**self).position()
    }

    fn frames_pending(&self) -> usize {
        (**self).frames_pending()
    }

    fn frames_drop(&mut self) {
        (**self).frames_drop()
    }

    fn capabilities(&self) -> &TerminalCaps {
        (**self).capabilities()
    }
}

/// Terminal capabilities
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalCaps {
    pub depth: ColorDepth,
    pub glyphs: bool,
    pub kitty_keyboard: bool,
}

impl Default for TerminalCaps {
    fn default() -> Self {
        Self {
            depth: ColorDepth::EightBit,
            glyphs: false,
            kitty_keyboard: false,
        }
    }
}

pub struct TerminalDrain<'a>(&'a mut dyn Terminal);

impl Iterator for TerminalDrain<'_> {
    type Item = TerminalEvent;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.poll(Some(Duration::new(0, 0))).ok().flatten()
    }
}

/// Waker object
///
/// Waker object is a thread safe object that can be called to wake terminal
/// with TerminalEvent::Wake event
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

    /// Wake terminal
    pub fn wake(&self) -> Result<(), Error> {
        (self.wake)()
    }
}

/// Object returned by handler function inside run method.
pub enum TerminalAction<R> {
    /// Quit run method with result `R`
    Quit(R),
    /// Wait for the next event from terminal
    Wait,
    /// Wait for the next event and do not render current frame
    WaitNoFrame,
    /// Wait for the next event with the provided timeout
    Sleep(Duration),
}

/// Commands that can be executed by terminal
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum TerminalCommand {
    /// Put character
    Char(char),
    /// Set current face (foreground/background colors and text attributes)
    Face(Face),
    /// Modify current face
    FaceModify(FaceModify),
    /// Get current face
    FaceGet,
    /// Control specified DEC mode (DECSET|DECRST)
    DecModeSet { enable: bool, mode: DecMode },
    /// Report specified DEC mode (DECRQM)
    DecModeGet(DecMode),
    /// Request current cursor position
    CursorGet,
    /// Move cursor to specified row and column
    CursorTo(Position),
    /// Move cursor to relative row and column to the current position
    CursorMove { row: i32, col: i32 },
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
    /// Erase entire screen
    EraseScreen,
    /// Erase specified amount of characters to the right from current cursor position
    EraseChars(usize),
    /// Scroll, positive is up and negative is down
    Scroll(i32),
    /// Set scroll region
    ScrollRegion { start: usize, end: usize },
    /// Full reset of the terminal
    Reset,
    /// Draw image
    Image(Image, Position),
    /// Erase image
    ImageErase(Image, Option<Position>),
    /// Request Termcap/Terminfo XTGETTCAP
    Termcap(Vec<String>),
    /// Set or query terminal colors
    Color {
        name: TerminalColor,
        color: Option<RGBA>,
    },
    /// Set terminal title
    Title(String),
    /// [Primary Device Attributes](https://vt100.net/docs/vt510-rm/DA1.html)
    DeviceAttrs,
    /// Set kitty keyboard protocol level
    KeyboardLevel(usize),
    /// Write raw data to terminal
    Raw(Vec<u8>),
}

impl TerminalCommand {
    /// Create alternative screen set command
    pub fn altscreen_set(enable: bool) -> Self {
        TerminalCommand::DecModeSet {
            enable,
            mode: DecMode::AltScreen,
        }
    }

    /// Create visible cursor set command
    pub fn visible_cursor_set(enable: bool) -> Self {
        TerminalCommand::DecModeSet {
            enable,
            mode: DecMode::VisibleCursor,
        }
    }

    /// Create auto wrap set command
    pub fn auto_wrap_set(enable: bool) -> Self {
        TerminalCommand::DecModeSet {
            enable,
            mode: DecMode::AutoWrap,
        }
    }

    /// Enable or disable mouse events
    pub fn mouse_events_set(
        enable: bool,
        motion: bool,
    ) -> impl IntoIterator<Item = TerminalCommand> {
        if enable {
            let motion_mode = if motion {
                TerminalCommand::DecModeSet {
                    enable: motion,
                    mode: DecMode::MouseMotions,
                }
            } else {
                TerminalCommand::DecModeSet {
                    enable: true,
                    mode: DecMode::MouseReport,
                }
            };
            Either::Left(
                [
                    motion_mode,
                    TerminalCommand::DecModeSet {
                        enable: true,
                        mode: DecMode::MouseSGR,
                    },
                ]
                .into_iter(),
            )
        } else {
            Either::Right(
                [
                    TerminalCommand::DecModeSet {
                        enable: false,
                        mode: DecMode::MouseSGR,
                    },
                    TerminalCommand::DecModeSet {
                        enable: false,
                        mode: DecMode::MouseReport,
                    },
                    TerminalCommand::DecModeSet {
                        enable: false,
                        mode: DecMode::MouseMotions,
                    },
                ]
                .into_iter(),
            )
        }
    }
}

/// Kind of terminal color
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TerminalColor {
    Background,
    Foreground,
    Palette(usize),
}

/// DEC mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DecMode {
    /// Visibility of the cursor
    VisibleCursor = 25,
    /// Wrapping of the text when it reaches end of the line
    AutoWrap = 7,
    /// Sixel scrolling
    SixelScrolling = 80,
    /// Enable/Disable mouse reporting
    MouseReport = 1000,
    /// Report mouse motion events if `MouseReport` is enabled
    MouseMotions = 1003,
    /// Report mouse event in SGR format
    MouseSGR = 1006,
    /// Alternative screen mode
    AltScreen = 1049,
    /// Synchronized output <https://gist.github.com/christianparpart/d8a62cc1ab659194337d73e399004036>
    SynchronizedOutput = 2026,
    /// Bracketed paste
    BracketedPaste = 2004,
}

/// Current/requested position of terminal
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Position {
    /// Row
    pub row: usize,
    /// Column
    pub col: usize,
}

impl Position {
    #[inline]
    pub const fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }

    #[inline]
    pub const fn origin() -> Self {
        Self { row: 0, col: 0 }
    }
}

impl DecMode {
    /// Convert DEC code into DecMode object
    pub fn from_usize(code: usize) -> Option<Self> {
        use DecMode::*;
        for mode in [
            VisibleCursor,
            AutoWrap,
            SixelScrolling,
            MouseReport,
            MouseMotions,
            MouseSGR,
            AltScreen,
            SynchronizedOutput,
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

/// Dec mode status
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DecModeStatus {
    /// Unknown DEC mode
    NotRecognized = 0,
    /// DEC mode enabled
    Enabled = 1,
    /// DEC mode disabled
    Disabled = 2,
    /// DEC mode was understood but can not be disabled
    PermanentlyEnabled = 3,
    /// DEC mode was understood but can not be enabled
    PermanentlyDisabled = 4,
}

impl DecModeStatus {
    /// Convert DEC status code into DecModeStatus object
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

/// Events returned by terminal
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum TerminalEvent {
    /// Key press event
    Key(Key),
    /// Mouse event
    Mouse(Mouse),
    /// Current cursor position
    CursorPosition(Position),
    /// Terminal was resized
    Resize(TerminalSize),
    /// Current terminal size
    Size(TerminalSize),
    /// DEC mode status
    DecMode {
        mode: DecMode,
        status: DecModeStatus,
    },
    /// Kitty image result
    KittyImage {
        id: u64,
        placement: Option<u64>,
        error: Option<String>,
    },
    /// Kitty keyboard level
    KeyboardLevel(usize),
    /// Terminal have been woken by waker object
    Wake,
    /// Termcap/Terminfo response to XTGETTCAP
    Termcap(BTreeMap<String, Option<String>>),
    /// Terminal Attributes DA1 response
    DeviceAttrs(BTreeSet<usize>),
    /// Unrecognized bytes (TODO: remove Vec and just use u8)
    Raw(Vec<u8>),
    /// Color
    Color { name: TerminalColor, color: RGBA },
    /// Report current face
    FaceGet(Face),
    /// So we can use single decoder for commands and events
    Command(TerminalCommand),
    /// Bracketed paste mode
    Paste(String),
}

/// Size
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct Size {
    pub height: usize,
    pub width: usize,
}

impl Size {
    /// Zero size
    #[inline]
    pub const fn empty() -> Self {
        Self::new(0, 0)
    }

    /// Create new size
    #[inline]
    pub const fn new(height: usize, width: usize) -> Self {
        Self { height, width }
    }

    /// Check if size is zero in any dimension
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.height * self.width == 0
    }

    /// Get size area
    #[inline]
    pub fn area(&self) -> usize {
        self.width * self.height
    }

    /// Expand/Contract size to bigger than `min` and smaller than `max` size
    pub fn clamp(self, min: Self, max: Self) -> Self {
        // NOTE: `self` is needed as reference will have lower precedence then Ord::clamp
        Size {
            height: self.height.clamp(min.height, max.height),
            width: self.width.clamp(min.width, max.width),
        }
    }
}

impl FromStr for Size {
    type Err = Error;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        let mut values = string.split([' ', ',']).map(|s| s.trim().parse().ok());
        let height = values
            .next()
            .flatten()
            .ok_or_else(|| Error::ParseError("Size", "failed to parse height".to_owned()))?;
        let width = values
            .next()
            .flatten()
            .ok_or_else(|| Error::ParseError("Size", "failed to parse width".to_owned()))?;
        Ok(Size { height, width })
    }
}

impl fmt::Display for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.height, self.width)
    }
}

/// Terminal size object
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TerminalSize {
    /// Size of the terminal in cells
    pub cells: Size,
    /// Size of the terminal in pixels
    pub pixels: Size,
}

impl TerminalSize {
    /// Size of the cell in pixels
    pub fn pixels_per_cell(&self) -> Size {
        Size {
            height: if self.cells.height > 0 {
                self.pixels.height / self.cells.height
            } else {
                0
            },
            width: if self.cells.width > 0 {
                self.pixels.width / self.cells.width
            } else {
                0
            },
        }
    }

    /// Convert cell size into pixels
    pub fn cells_in_pixels(&self, cells: Size) -> Size {
        let cell_size = self.pixels_per_cell();
        Size {
            height: cells.height * cell_size.height,
            width: cells.width * cell_size.width,
        }
    }
}

/// Terminal statistics
#[derive(Clone, Debug)]
pub struct TerminalStats {
    /// Number of bytes send
    pub send: usize,
    /// Number of bytes received
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

/// Mouse event
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Mouse {
    /// Key name
    pub name: KeyName,
    /// Key mode
    pub mode: KeyMod,
    /// Position
    pub pos: Position,
}

impl fmt::Debug for Mouse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.mode.is_empty() {
            write!(f, "{:?} [{},{}]", self.name, self.pos.row, self.pos.col)?;
        } else {
            write!(
                f,
                "{:?}+{:?} [{},{}]",
                self.name, self.mode, self.pos.row, self.pos.col
            )?;
        }
        Ok(())
    }
}
