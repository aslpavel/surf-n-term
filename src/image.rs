use crate::{
    common::clamp, encoder::base64_encode, Color, Error, Position, Shape, Surface, SurfaceMut,
    SurfaceOwned, Terminal, TerminalEvent, RGBA,
};
use std::{
    collections::HashMap,
    fmt,
    fmt::Write as _,
    io::{Cursor, Read, Write},
    ops::{Add, AddAssign, Mul},
    sync::Arc,
};

/// Arc wrapped RGBA surface with precomputed hash
#[derive(Clone)]
pub struct Image {
    surf: Arc<dyn Surface<Item = RGBA>>,
    hash: u64,
}

impl Image {
    pub fn new(surf: impl Surface<Item = RGBA> + 'static) -> Self {
        Self {
            hash: surf.hash(),
            surf: Arc::new(surf),
        }
    }

    /// Image size in bytes
    pub fn size(&self) -> usize {
        self.surf.height() * self.surf.width() * 4
    }
}

impl PartialEq for Image {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
    }
}

impl Eq for Image {}

impl fmt::Debug for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Image({})", self.hash)
    }
}

impl Surface for Image {
    type Item = RGBA;

    fn shape(&self) -> Shape {
        self.surf.shape()
    }

    fn hash(&self) -> u64 {
        self.hash
    }

    fn data(&self) -> &[Self::Item] {
        self.surf.data()
    }
}

pub trait ImageHandler {
    /// Name
    fn name(&self) -> &str;

    /// Draw image
    ///
    /// Send approprite terminal escape sequence so image would be rendered.
    fn draw(&mut self, img: &Image, out: &mut dyn Write) -> Result<(), Error>;

    /// Erase image at specified position
    ///
    /// This is needed when erasing characters is not actually removing
    /// image from the terminal. For example kitty needs to send separate
    /// escape sequence to actually erase image.
    fn erase(&mut self, pos: Position, out: &mut dyn Write) -> Result<(), Error>;

    /// Handle events frome the terminal
    ///
    /// True means event has been handled and should not be propagated to a user
    fn handle(&mut self, event: &TerminalEvent) -> Result<bool, Error>;
}

/// Detect appropriate image handler for provided termainl
pub fn image_handler_detect(
    _term: &mut dyn Terminal,
) -> Result<Option<Box<dyn ImageHandler>>, Error> {
    Ok(Some(Box::new(KittyImageHandler::new(true))))
}

/// Image handler for kitty graphic protocol
///
/// Reference: https://sw.kovidgoyal.net/kitty/graphics-protocol.html
pub struct KittyImageHandler {
    imgs: HashMap<u64, usize>, // hash -> size in bytes
    cache: bool,
}

impl KittyImageHandler {
    pub fn new(cache: bool) -> Self {
        Self {
            imgs: Default::default(),
            cache,
        }
    }
}

impl Default for KittyImageHandler {
    fn default() -> Self {
        Self::new(true)
    }
}

impl ImageHandler for KittyImageHandler {
    fn name(&self) -> &str {
        "kitty"
    }

    fn draw(&mut self, img: &Image, out: &mut dyn Write) -> Result<(), Error> {
        let id = img.hash() % 4294967295;
        match self.imgs.get(&id) {
            Some(_) => {
                // data has already been send and we can just use an id.
                write!(out, "\x1b_Ga=p,i={};\x1b\\", id)?;
            }
            None => {
                let raw: Vec<_> = img.iter().flat_map(|c| RGBAIter::new(*c)).collect();
                // TODO: stream into base64_encode
                let compressed =
                    miniz_oxide::deflate::compress_to_vec_zlib(&raw, /* level */ 5);
                let mut base64 = Cursor::new(Vec::new());
                base64_encode(base64.get_mut(), compressed.iter().copied())?;

                let mut buf = [0u8; 4096];
                loop {
                    let size = base64.read(&mut buf)?;
                    let more = if base64.position() < base64.get_ref().len() as u64 {
                        1
                    } else {
                        0
                    };
                    // a = t - transfer only
                    // a = T - transfer and show
                    // a = p - present only using `i = id`
                    write!(
                        out,
                        "\x1b_Ga=T,f=32,o=z,v={},s={},m={}",
                        img.height(),
                        img.width(),
                        more,
                    )?;
                    if self.cache {
                        write!(out, "i={}", id)?;
                    }
                    out.write_all(b";")?;
                    out.write_all(&buf[..size])?;
                    out.write_all(b"\x1b\\")?;
                    if more == 0 {
                        break;
                    }
                }
                if self.cache {
                    self.imgs.insert(id, img.size());
                }
            }
        }
        Ok(())
    }

    fn erase(&mut self, pos: Position, out: &mut dyn Write) -> Result<(), Error> {
        write!(
            out,
            "\x1b_Ga=d,d=p,x={},y={}\x1b\\",
            pos.col + 1,
            pos.row + 1
        )?;
        Ok(())
    }

    fn handle(&mut self, event: &TerminalEvent) -> Result<bool, Error> {
        // TODO:
        //   - we should probably resend image again if it failed to draw by id (pushed out of cache)
        //   - probably means we should track where image is supposed to be drawn or whole frame should
        //     be re-drawn
        match event {
            TerminalEvent::KittyImage { .. } => Ok(true),
            _ => Ok(false),
        }
    }
}

struct RGBAIter {
    color: [u8; 4],
    index: usize,
}

impl RGBAIter {
    fn new(color: RGBA) -> Self {
        Self {
            color: color.rgba_u8(),
            index: 0,
        }
    }
}

impl Iterator for RGBAIter {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.color.get(self.index).copied();
        self.index += 1;
        result
    }
}

/// Qunatize image
///
/// Performs palette extraction with `OcTree` and apply Floyd–Steinberg dithering.
pub fn quantize_and_dither(
    img: impl Surface<Item = RGBA>,
    palette_size: usize,
) -> (Vec<RGBA>, SurfaceOwned<usize>) {
    // build and prune octree unitil desired size
    let mut octree = OcTree::from_surface(&img);
    octree.prune_until(palette_size);

    // allocate data
    let mut color_to_index: HashMap<RGBA, usize> = HashMap::new();
    let mut palette: Vec<RGBA> = Vec::new();
    let mut qimg = SurfaceOwned::new(img.height(), img.width());

    // quantize and dither
    let mut errors: Vec<ColorError> = Vec::new();
    let ewidth = img.width() + 2; // to evaoid check for first and the last pixels
    errors.resize_with(ewidth * 2, || ColorError::new());
    for row in 0..img.height() {
        // swap error rows
        for col in 0..ewidth + 2 {
            errors[col] = errors[col + ewidth];
            errors[col + ewidth] = ColorError::new();
        }
        // qunatize and spread the error
        for col in 0..img.width() {
            let color = img.get(row, col).unwrap().clone();
            let color = errors[col + 1].apply(color); // account for error
            let qcolor = octree
                .find(color)
                .expect("[quantize] octree cannot find color")
                .clone();

            // allocate palette index
            let index = match color_to_index.get(&qcolor) {
                Some(index) => *index,
                None => {
                    let index = palette.len();
                    color_to_index.insert(qcolor, index);
                    palette.push(qcolor);
                    index
                }
            };
            qimg.set(row, col, index);
            // spread the error according to Floyd–Steinberg dithering matrix:
            // [[0   , X   , 7/16],
            // [3/16, 5/16, 1/16]]
            let error = ColorError::between(color, qcolor);
            errors[col + 2] += error * 0.4375; // 7/16
            errors[col + ewidth - 1] += error * 0.1875; // 3/16
            errors[col + ewidth] += error * 0.3125; // 5/16
            errors[col + ewidth + 1] += error * 0.0625; // 1/16
        }
    }

    (palette, qimg)
}

#[derive(Clone, Copy)]
pub struct ColorError([f32; 3]);

impl ColorError {
    fn new() -> Self {
        Self([0.0; 3])
    }

    fn between(c0: RGBA, c1: RGBA) -> Self {
        let [r0, g0, b0] = c0.rgb_u8();
        let [r1, g1, b1] = c1.rgb_u8();
        Self([
            r0 as f32 - r1 as f32,
            g0 as f32 - g1 as f32,
            b0 as f32 - b1 as f32,
        ])
    }

    fn apply(self, color: RGBA) -> RGBA {
        let [r, g, b] = color.rgb_u8();
        let Self([re, ge, be]) = self;
        RGBA::new(
            clamp(r as f32 + re, 0.0, 255.0).round() as u8,
            clamp(g as f32 + ge, 0.0, 255.0).round() as u8,
            clamp(b as f32 + be, 0.0, 255.0).round() as u8,
            255,
        )
    }
}

impl Add<Self> for ColorError {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self::Output {
        let Self([r0, g0, b0]) = self;
        let Self([r1, g1, b1]) = other;
        Self([r0 + r1, g0 + g1, b0 + b1])
    }
}

impl AddAssign for ColorError {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}

impl Mul<f32> for ColorError {
    type Output = Self;

    #[inline]
    fn mul(self, val: f32) -> Self::Output {
        let Self([r, g, b]) = self;
        Self([r * val, g * val, b * val])
    }
}

#[derive(Debug, Clone, Copy)]
struct OcTreeLeaf {
    red_acc: usize,
    green_acc: usize,
    blue_acc: usize,
    color_count: usize,
}

impl OcTreeLeaf {
    fn new() -> Self {
        Self {
            red_acc: 0,
            green_acc: 0,
            blue_acc: 0,
            color_count: 0,
        }
    }
}

#[derive(Debug, Clone)]
enum OcTreeNode {
    Leaf(OcTreeLeaf),
    Tree(Box<OcTree>),
    Empty,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OcTreeStats {
    pub leaf_count: usize,
    pub min_color: Option<usize>,
}

impl OcTreeStats {
    pub fn new(leaf_count: usize, min_color: Option<usize>) -> Self {
        Self {
            leaf_count,
            min_color,
        }
    }

    // Monoidal unit
    pub fn empty() -> Self {
        Self::new(0, None)
    }

    // Monoidal sum
    pub fn join(self, other: Self) -> Self {
        Self {
            leaf_count: self.leaf_count + other.leaf_count,
            min_color: match (self.min_color, other.min_color) {
                (Some(c0), Some(c1)) => Some(std::cmp::min(c0, c1)),
                (None, Some(c1)) => Some(c1),
                (Some(c0), None) => Some(c0),
                (None, None) => None,
            },
        }
    }

    // Monodial sum over oll stats of nodes in the slice
    fn from_slice(slice: &[OcTreeNode]) -> Self {
        slice
            .iter()
            .fold(Self::empty(), |acc, n| acc.join(n.stats()))
    }
}

impl OcTreeNode {
    fn take(&mut self) -> Self {
        std::mem::replace(self, Self::Empty)
    }

    fn stats(&self) -> OcTreeStats {
        use OcTreeNode::*;
        match self {
            Empty => OcTreeStats::empty(),
            Leaf(leaf) => OcTreeStats {
                leaf_count: 1,
                min_color: Some(leaf.color_count),
            },
            Tree(tree) => tree.stats,
        }
    }
}

/// Oc(tet)Tree use for color quantization
///
/// Reference: https://www.cubic.org/docs/octree.htm
#[derive(Debug, Clone)]
pub struct OcTree {
    stats: OcTreeStats,
    children: [OcTreeNode; 8],
}

impl OcTree {
    pub fn new() -> Self {
        use OcTreeNode::Empty;
        Self {
            stats: OcTreeStats::empty(),
            children: [Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty],
        }
    }

    pub fn stats(&self) -> OcTreeStats {
        self.stats
    }

    pub fn from_surface<S, C>(surf: S) -> Self
    where
        S: Surface<Item = C>,
        C: Color,
    {
        let mut tree = OcTree::new();
        for color in surf.iter().copied() {
            tree.insert(color)
        }
        tree
    }

    /// Find nearest color inside the octree
    pub fn find(&self, color: impl Color) -> Option<RGBA> {
        use OcTreeNode::*;
        let mut tree = self;
        for index in OcTreePath::new(color.rgb_u8()) {
            match &tree.children[index] {
                Empty => break,
                Leaf(leaf) => {
                    let r = (leaf.red_acc / leaf.color_count) as u8;
                    let g = (leaf.green_acc / leaf.color_count) as u8;
                    let b = (leaf.blue_acc / leaf.color_count) as u8;
                    return Some(RGBA::new(r, g, b, 255));
                }
                Tree(next_tree) => tree = next_tree,
            }
        }
        None
    }

    /// Take node (Note: statistics is broken at this point)
    fn node_take(&mut self, index: usize) -> OcTreeNode {
        self.children[index].take()
    }

    /// Update node with provided function preseving node tree stats
    fn node_update(&mut self, index: usize, func: impl FnOnce(OcTreeNode) -> OcTreeNode) {
        self.children[index] = func(self.children[index].take());
        self.stats = OcTreeStats::from_slice(&self.children);
    }

    /// Insert color into the octree
    pub fn insert(&mut self, color: impl Color) {
        let mut path = OcTreePath::new(color.rgb_u8());
        let index = path.next().expect("OcTreePath can not be empty");
        self.node_update(index, |node| Self::insert_rec(node, path));
    }

    /// Recursive insertion of the color into a node
    ///
    /// Returns updated node and number of leafs added.
    fn insert_rec(mut node: OcTreeNode, mut path: OcTreePath) -> OcTreeNode {
        use OcTreeNode::*;
        match path.next() {
            Some(index) => match node {
                Empty => {
                    let mut tree = Self::new();
                    tree.node_update(index, move |node| Self::insert_rec(node, path));
                    Tree(Box::new(tree))
                }
                Leaf(mut leaf) => {
                    let [r, g, b] = path.rgb();
                    leaf.red_acc += r as usize;
                    leaf.green_acc += g as usize;
                    leaf.blue_acc += b as usize;
                    leaf.color_count += 1;
                    node
                }
                Tree(ref mut tree) => {
                    tree.node_update(index, move |node| Self::insert_rec(node, path));
                    node
                }
            },
            None => match node {
                Empty => {
                    let [r, g, b] = path.rgb();
                    let leaf = OcTreeLeaf {
                        red_acc: r as usize,
                        green_acc: g as usize,
                        blue_acc: b as usize,
                        color_count: 1,
                    };
                    Leaf(leaf)
                }
                Leaf(mut leaf) => {
                    let [r, g, b] = path.rgb();
                    leaf.red_acc += r as usize;
                    leaf.green_acc += g as usize;
                    leaf.blue_acc += b as usize;
                    leaf.color_count += 1;
                    Leaf(leaf)
                }
                Tree(_) => unreachable!(),
            },
        }
    }

    /// Prune until desired number of colors is left
    pub fn prune_until(&mut self, color_count: usize) {
        while self.stats.leaf_count > color_count {
            self.prune()
        }
    }

    /// Prune octree
    ///
    /// Search the node, where the sum of the childs references is minimal and
    /// convert it to a leaf.
    pub fn prune(&mut self) {
        if let Some(index) = self.find_argmin_min_color() {
            self.node_update(index, Self::prune_rec)
        }
    }

    /// Recursive pruning of the sub-tree
    fn prune_rec(node: OcTreeNode) -> OcTreeNode {
        use OcTreeNode::*;
        let mut tree = match node {
            Tree(tree) => tree,
            _ => return node,
        };
        match tree.find_argmin_min_color() {
            None => {
                // we have found the node with minimal number of refernces
                let mut leaf = OcTreeLeaf::new();
                for index in 0..8 {
                    match tree.node_take(index) {
                        Tree(_) => unreachable!("OcTree::find_count_argmin failed to find subtree"),
                        Empty => continue,
                        Leaf(other) => {
                            leaf.red_acc += other.red_acc;
                            leaf.green_acc += other.green_acc;
                            leaf.blue_acc += other.blue_acc;
                            leaf.color_count += other.color_count;
                        }
                    }
                }
                Leaf(leaf)
            }
            Some(index) => {
                tree.node_update(index, Self::prune_rec);
                Tree(tree)
            }
        }
    }

    /// Find **sub-tree** with minimum count of colors
    fn find_argmin_min_color(&self) -> Option<usize> {
        let mut min = None;
        let mut argmin = None;
        for index in 0..8 {
            let min_new = match &self.children[index] {
                OcTreeNode::Tree(tree) => tree.stats.min_color,
                _ => continue,
            };
            match min {
                None => {
                    min = Some(min_new);
                    argmin = Some(index);
                }
                Some(min_old) if min_new < min_old => {
                    min = Some(min_new);
                    argmin = Some(index);
                }
                _ => (),
            }
        }
        argmin
    }

    pub fn debug_palette(&self) {
        use OcTreeNode::*;
        for child in self.children.iter() {
            match child {
                Empty => continue,
                Leaf(leaf) => {
                    let r = (leaf.red_acc / leaf.color_count) as u8;
                    let g = (leaf.green_acc / leaf.color_count) as u8;
                    let b = (leaf.blue_acc / leaf.color_count) as u8;
                    print!("\x1b[48;2;{};{};{}m  \x1b[m", r, g, b);
                }
                Tree(tree) => tree.debug_palette(),
            }
        }
    }

    pub fn dot(&self) -> Result<String, fmt::Error> {
        let mut next = 1;
        let mut out = String::new();
        writeln!(&mut out, "digraph OcTree {{")?;
        writeln!(&mut out, "  rankdir=\"LR\"")?;
        writeln!(
            &mut out,
            "  0 [label=\"{} [{}]\"]",
            self.stats.min_color.unwrap_or(0),
            self.stats.leaf_count,
        )?;
        self.dot_rec(0, &mut next, &mut out)?;
        writeln!(&mut out, "}}")?;
        Ok(out)
    }

    pub fn dot_rec(
        &self,
        parent: usize,
        next: &mut usize,
        out: &mut dyn std::fmt::Write,
    ) -> fmt::Result {
        use OcTreeNode::*;

        for child in self.children.iter() {
            match child {
                Empty => continue,
                Leaf(leaf) => {
                    let r = (leaf.red_acc / leaf.color_count) as u8;
                    let g = (leaf.green_acc / leaf.color_count) as u8;
                    let b = (leaf.blue_acc / leaf.color_count) as u8;
                    let color = RGBA::new(r, g, b, 255);

                    let id = *next;
                    *next += 1;

                    writeln!(
                        out,
                        "  {} [style=filled, color=\"{:?}\", label=\"{}\"]",
                        id, color, leaf.color_count
                    )?;
                    writeln!(out, "  {} -> {}", parent, id)?;
                }
                Tree(tree) => {
                    let id = *next;
                    *next += 1;

                    let stats = tree.stats();
                    writeln!(
                        out,
                        "  {} [label=\"{} [{}]\"]",
                        id,
                        stats.min_color.unwrap_or(0),
                        stats.leaf_count,
                    )?;
                    writeln!(out, "  {} -> {}", parent, id)?;
                    tree.dot_rec(id, next, out)?
                }
            }
        }

        Ok(())
    }
}

/// Iterator which goes over all most significant bits of the color
/// concatinated together.
///
/// Example:
/// For RGB (90, 13, 157) in binary form
/// R 0 1 0 1 1 0 1 0
/// G 0 1 1 1 0 0 0 1
/// B 1 0 0 1 1 1 0 1
/// Output will be [0b001, 0b110, 0b010, 0b111, 0b101, 0b001, 0b100, 0b011]
struct OcTreePath {
    rgb: [u8; 3],
    state: u32,
    length: u8,
}

impl OcTreePath {
    pub fn new(rgb: [u8; 3]) -> Self {
        let [r, g, b] = rgb;
        // pack RGB components into u32 value
        let state = ((r as u32) << 16) | ((g as u32) << 8) | b as u32;
        Self {
            rgb,
            state,
            length: 8,
        }
    }

    pub fn rgb(&self) -> [u8; 3] {
        self.rgb
    }
}

impl Iterator for OcTreePath {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.length == 0 {
            return None;
        }
        self.length -= 1;
        // - We should pick most significant bit from each compenent
        //   and concatinate into one value to get an index inside
        //   octree.
        // - Left shift all components and set least significant bits
        //   of all components to zero.
        // - Repeat until all bits of all components are zero
        let bits = self.state & 0x00808080;
        self.state = (self.state << 1) & 0x00fefefe;
        let value = ((bits >> 21) | (bits >> 14) | (bits >> 7)) & 0b111;
        Some(value as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_octree_path() -> Result<(), Error> {
        let path: Vec<_> = OcTreePath::new("#5a719d".parse::<RGBA>()?.rgb_u8()).collect();
        assert_eq!(path, vec![1, 6, 2, 7, 5, 1, 4, 3]);

        let path: Vec<_> = OcTreePath::new("#808080".parse::<RGBA>()?.rgb_u8()).collect();
        assert_eq!(path, vec![7, 0, 0, 0, 0, 0, 0, 0]);

        let path: Vec<_> = OcTreePath::new("#d3869b".parse::<RGBA>()?.rgb_u8()).collect();
        assert_eq!(path, vec![7, 4, 0, 5, 1, 2, 7, 5]);

        Ok(())
    }

    #[test]
    fn test_octree_stats() {
        let s0 = OcTreeStats::new(1, Some(1));
        let s1 = OcTreeStats::new(2, Some(1));
        let s2 = OcTreeStats::new(2, Some(2));
        let s3 = OcTreeStats::new(1, None);
        assert_eq!(s0.join(s1), OcTreeStats::new(3, Some(1)));
        assert_eq!(s0.join(s2), OcTreeStats::new(3, Some(1)));
        assert_eq!(s0.join(s3), OcTreeStats::new(2, Some(1)));

        let mut tree = OcTree::new();
        tree.node_update(1, |_| {
            OcTreeNode::Leaf(OcTreeLeaf {
                red_acc: 1,
                green_acc: 2,
                blue_acc: 3,
                color_count: 4,
            })
        });
        assert_eq!(tree.stats, OcTreeStats::new(1, Some(4)));
    }

    #[test]
    fn test_octree() -> Result<(), Error> {
        let c0 = "#5a719d".parse::<RGBA>()?;
        let c1 = "#d3869b".parse::<RGBA>()?;

        let mut tree = OcTree::new();

        tree.insert(c0);
        tree.insert(c0);
        assert_eq!(tree.stats(), OcTreeStats::new(1, Some(2)));
        assert_eq!(tree.find_argmin_min_color(), Some(1));
        assert_eq!(tree.find(c0), Some(c0));

        tree.insert(c1);
        assert_eq!(tree.stats(), OcTreeStats::new(2, Some(1)));
        assert_eq!(tree.find_argmin_min_color(), Some(7));
        assert_eq!(tree.find(c1), Some(c1));

        Ok(())
    }
}
