use crate::{
    common::{clamp, Rnd},
    encoder::base64_encode,
    Color, Error, Position, Shape, Surface, SurfaceMut, SurfaceOwned, Terminal, TerminalEvent,
    RGBA,
};
use std::{
    collections::HashMap,
    fmt,
    io::{Cursor, Read, Write},
    iter::FromIterator,
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
/// Performs palette extraction with `OcTreeQuantizer` and apply Floyd–Steinberg dithering.
pub fn quantize_and_dither(
    img: impl Surface<Item = RGBA>,
    palette_size: usize,
) -> Option<(ColorPalette, SurfaceOwned<usize>)> {
    let mut quantizer = OcTreeQuantizer::new(&img, palette_size)?;
    let mut qimg = SurfaceOwned::new(img.height(), img.width());

    // quantize and dither
    let mut errors: Vec<ColorError> = Vec::new();
    let ewidth = img.width() + 2; // to evaoid check for first and the last pixels
    errors.resize_with(ewidth * 2, || ColorError::new());
    for row in 0..img.height() {
        // swap error rows
        for col in 0..ewidth {
            errors[col] = errors[col + ewidth];
            errors[col + ewidth] = ColorError::new();
        }
        // qunatize and spread the error
        for col in 0..img.width() {
            let color = img.get(row, col).unwrap().clone();
            let color = errors[col + 1].add(color); // account for error
            let (qindex, qcolor) = quantizer.quantize(color);
            qimg.set(row, col, qindex);
            // spread the error according to Floyd–Steinberg dithering matrix:
            // [[0   , X   , 7/16],
            // [3/16, 5/16, 1/16]]
            let error = ColorError::between(color, qcolor);
            errors[col + 2] += error * 0.4375; // 7/16
            errors[col + ewidth] += error * 0.1875; // 3/16
            errors[col + ewidth + 1] += error * 0.3125; // 5/16
            errors[col + ewidth + 2] += error * 0.0625; // 1/16
        }
    }

    Some((quantizer.into_palette(), qimg))
}

#[derive(Clone, Copy)]
pub struct ColorError([f32; 3]);

impl ColorError {
    fn new() -> Self {
        Self([0.0; 3])
    }

    /// Error between two colors
    fn between(c0: RGBA, c1: RGBA) -> Self {
        let [r0, g0, b0] = c0.rgb_u8();
        let [r1, g1, b1] = c1.rgb_u8();
        Self([
            r0 as f32 - r1 as f32,
            g0 as f32 - g1 as f32,
            b0 as f32 - b1 as f32,
        ])
    }

    /// Add error to the color
    fn add(self, color: RGBA) -> RGBA {
        let [r, g, b] = color.rgb_u8();
        let Self([re, ge, be]) = self;
        RGBA::new(
            clamp(r as f32 + re, 0.0, 255.0) as u8,
            clamp(g as f32 + ge, 0.0, 255.0) as u8,
            clamp(b as f32 + be, 0.0, 255.0) as u8,
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

pub struct OcTreeQuantizer {
    octree: OcTree,
    palette: ColorPalette,
}

impl OcTreeQuantizer {
    pub fn new(img: impl Surface<Item = RGBA>, palette_size: usize) -> Option<Self> {
        let sample: u32 = (img.height() * img.width() / (palette_size * 100)) as u32;
        let mut octree: OcTree = if sample < 2 {
            img.iter().copied().collect()
        } else {
            let mut octree = OcTree::new();
            let mut rnd = Rnd::new(5);
            let mut colors = img.iter().copied();
            while let Some(color) = colors.nth((rnd.next_u32() % sample) as usize) {
                octree.insert(color);
            }
            octree
        };
        octree.prune_until(palette_size);
        // octree.prune_empty();
        // octree.prune_until(palette_size);

        let palette = octree.build_palette()?;
        Some(Self { octree, palette })
    }

    pub fn quantize(&mut self, color: RGBA) -> (usize, RGBA) {
        match self.octree.find(color) {
            Some(result) => result,
            None => self.palette.find(color),
        }
    }

    pub fn into_palette(self) -> ColorPalette {
        self.palette
    }
}

#[derive(Debug, Clone, Copy)]
struct OcTreeLeaf {
    red_acc: usize,
    green_acc: usize,
    blue_acc: usize,
    color_count: usize,
    index: usize,
}

impl OcTreeLeaf {
    fn new() -> Self {
        Self {
            red_acc: 0,
            green_acc: 0,
            blue_acc: 0,
            color_count: 0,
            index: 0,
        }
    }

    fn from_rgba(rgba: RGBA) -> Self {
        let [r, g, b] = rgba.rgb_u8();
        Self {
            red_acc: r as usize,
            green_acc: g as usize,
            blue_acc: b as usize,
            color_count: 1,
            index: 0,
        }
    }

    fn to_rgba(&self) -> RGBA {
        let r = (self.red_acc / self.color_count) as u8;
        let g = (self.green_acc / self.color_count) as u8;
        let b = (self.blue_acc / self.color_count) as u8;
        RGBA::new(r, g, b, 255)
    }
}

impl AddAssign<RGBA> for OcTreeLeaf {
    fn add_assign(&mut self, rhs: RGBA) {
        let [r, g, b] = rhs.rgb_u8();
        self.red_acc += r as usize;
        self.green_acc += g as usize;
        self.blue_acc += b as usize;
        self.color_count += 1;
    }
}

impl AddAssign<OcTreeLeaf> for OcTreeLeaf {
    fn add_assign(&mut self, rhs: Self) {
        self.red_acc += rhs.red_acc;
        self.green_acc += rhs.green_acc;
        self.blue_acc += rhs.blue_acc;
        self.color_count += rhs.color_count;
    }
}

#[derive(Debug, Clone)]
enum OcTreeNode {
    Leaf(OcTreeLeaf),
    Tree(Box<OcTree>),
    Empty,
}

impl OcTreeNode {
    pub fn is_empty(&self) -> bool {
        match self {
            OcTreeNode::Empty => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OcTreeInfo {
    // total number of leafs in the subtree
    pub leaf_count: usize,
    // total number of colors in the subtree
    pub color_count: usize,
    // node (Tree|Leaf) with smallest number of colors in the subtree
    pub min_color_count: Option<usize>,
}

impl OcTreeInfo {
    // Monoidal unit
    pub fn empty() -> Self {
        Self {
            leaf_count: 0,
            color_count: 0,
            min_color_count: None,
        }
    }

    // Monoidal sum
    pub fn join(self, other: Self) -> Self {
        let leaf_count = self.leaf_count + other.leaf_count;
        let color_count = self.color_count + other.color_count;
        let min_color_count = match (self.min_color_count, other.min_color_count) {
            (Some(c0), Some(c1)) => Some(std::cmp::min(c0, c1)),
            (None, Some(c1)) => Some(c1),
            (Some(c0), None) => Some(c0),
            (None, None) => None,
        };
        Self {
            leaf_count,
            color_count,
            min_color_count,
        }
    }

    // Monodial sum over oll infos of nodes in the slice
    fn from_slice(slice: &[OcTreeNode]) -> Self {
        slice
            .iter()
            .fold(Self::empty(), |acc, n| acc.join(n.info()))
    }
}

impl OcTreeNode {
    // Take node content and replace it with empty node
    fn take(&mut self) -> Self {
        std::mem::replace(self, Self::Empty)
    }

    // Get info associated with the node
    fn info(&self) -> OcTreeInfo {
        use OcTreeNode::*;
        match self {
            Empty => OcTreeInfo::empty(),
            Leaf(leaf) => OcTreeInfo {
                leaf_count: 1,
                color_count: leaf.color_count,
                min_color_count: Some(leaf.color_count),
            },
            Tree(tree) => tree.info,
        }
    }
}

/// Oc(tet)Tree use for color quantization
///
/// References:
/// - https://www.cubic.org/docs/octree.htm
/// - http://www.leptonica.org/color-quantization.html
#[derive(Debug, Clone)]
pub struct OcTree {
    info: OcTreeInfo,
    removed: OcTreeLeaf,
    children: [OcTreeNode; 8],
}

impl Extend<RGBA> for OcTree {
    fn extend<T: IntoIterator<Item = RGBA>>(&mut self, colors: T) {
        for color in colors {
            self.insert(color)
        }
    }
}

impl FromIterator<RGBA> for OcTree {
    fn from_iter<T: IntoIterator<Item = RGBA>>(iter: T) -> Self {
        let mut octree = OcTree::new();
        octree.extend(iter);
        octree
    }
}

impl OcTree {
    pub fn new() -> Self {
        use OcTreeNode::Empty;
        Self {
            info: OcTreeInfo::empty(),
            removed: OcTreeLeaf::new(),
            children: [Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty],
        }
    }

    pub fn info(&self) -> OcTreeInfo {
        self.info
    }

    /// Find nearest color inside the octree
    /// NOTE: to get correct palette index call build_palette first.
    pub fn find(&self, color: RGBA) -> Option<(usize, RGBA)> {
        use OcTreeNode::*;
        let mut tree = self;
        for index in OcTreePath::new(color) {
            match &tree.children[index] {
                Empty => break,
                Leaf(leaf) => return Some((leaf.index, leaf.to_rgba())),
                Tree(next_tree) => tree = next_tree,
            }
        }
        None
    }

    /// All color present in the octree
    pub fn build_palette(&mut self) -> Option<ColorPalette> {
        fn palette_rec(node: &mut OcTreeNode, palette: &mut Vec<RGBA>) {
            use OcTreeNode::*;
            match node {
                Empty => return,
                Leaf(ref mut leaf) => {
                    leaf.index = palette.len();
                    palette.push(leaf.to_rgba());
                }
                Tree(tree) => {
                    for child in tree.children.iter_mut() {
                        palette_rec(child, palette)
                    }
                }
            }
        }

        let mut palette = Vec::new();
        for child in self.children.iter_mut() {
            palette_rec(child, &mut palette);
        }
        ColorPalette::new(palette)
    }

    /// Update node with provided function.
    fn node_update(&mut self, index: usize, func: impl FnOnce(OcTreeNode) -> OcTreeNode) {
        self.children[index] = func(self.children[index].take());
        self.info = OcTreeInfo::from_slice(&self.children);
    }

    /// Insert color into the octree
    pub fn insert(&mut self, color: RGBA) {
        // Recursive insertion of the color into a node
        fn insert_rec(node: OcTreeNode, mut path: OcTreePath) -> OcTreeNode {
            use OcTreeNode::*;
            match path.next() {
                Some(index) => match node {
                    Empty => {
                        let mut tree = OcTree::new();
                        tree.node_update(index, move |node| insert_rec(node, path));
                        Tree(Box::new(tree))
                    }
                    Leaf(mut leaf) => {
                        leaf += path.rgba();
                        Leaf(leaf)
                    }
                    Tree(mut tree) => {
                        tree.node_update(index, move |node| insert_rec(node, path));
                        Tree(tree)
                    }
                },
                None => match node {
                    Empty => Leaf(OcTreeLeaf::from_rgba(path.rgba())),
                    Leaf(mut leaf) => {
                        leaf += path.rgba();
                        Leaf(leaf)
                    }
                    Tree(_) => unreachable!(),
                },
            }
        }

        let mut path = OcTreePath::new(color);
        let index = path.next().expect("OcTreePath can not be empty");
        self.node_update(index, |node| insert_rec(node, path));
    }

    /// Replace all empty nodes with corresponding colors
    pub fn prune_empty(&mut self) {
        fn prune_empty_rec(node: OcTreeNode, path: &mut Vec<usize>) -> OcTreeNode {
            use OcTreeNode::*;
            match node {
                Empty => Leaf(OcTreeLeaf::from_rgba(color_from_path(path))),
                Leaf(_) => return node,
                Tree(mut tree) => {
                    for index in 0..8 {
                        path.push(index);
                        tree.children[index] = prune_empty_rec(tree.children[index].take(), path);
                        path.pop();
                    }
                    tree.info = OcTreeInfo::from_slice(&tree.children);
                    Tree(tree)
                }
            }
        }

        let mut path = Vec::with_capacity(8);
        for index in 0..8 {
            path.push(index);
            self.children[index] = prune_empty_rec(self.children[index].take(), &mut path);
            path.pop();
        }
        self.info = OcTreeInfo::from_slice(&self.children);
    }

    /// Prune until desired number of colors is left
    pub fn prune_until(&mut self, color_count: usize) {
        while self.info.leaf_count > color_count {
            self.prune();
        }
    }

    /// Remove the node with minimal number of colors in the it
    pub fn prune(&mut self) {
        use OcTreeNode::*;

        // find child index with minimal color count in the child subtree
        fn argmin_color_count(tree: &OcTree) -> Option<usize> {
            tree.children
                .iter()
                .enumerate()
                .filter_map(|(index, node)| Some((index, node.info().min_color_count?)))
                .min_by_key(|(_, min_tail_tree)| *min_tail_tree)
                .map(|(index, _)| index)
        }

        // recursive prune helper
        fn prune_rec(mut tree: Box<OcTree>) -> OcTreeNode {
            match argmin_color_count(&tree) {
                None => Leaf(tree.removed),
                Some(index) => match tree.children[index].take() {
                    Empty => unreachable!("agrmin_color_count found and empty node"),
                    Leaf(leaf) => {
                        tree.removed += leaf;
                        if tree.children.iter().all(OcTreeNode::is_empty) {
                            Leaf(tree.removed)
                        } else {
                            Tree(tree)
                        }
                    }
                    Tree(child_tree) => {
                        let child = prune_rec(child_tree);
                        match child {
                            Leaf(leaf) if tree.children.iter().all(OcTreeNode::is_empty) => {
                                tree.removed += leaf;
                                Leaf(tree.removed)
                            }
                            _ => {
                                tree.node_update(index, |_| child);
                                Tree(tree)
                            }
                        }
                    }
                },
            }
        }

        if let Some(index) = argmin_color_count(self) {
            match self.children[index].take() {
                Empty => unreachable!("agrmin_color_count found and empty node"),
                Leaf(leaf) => self.removed += leaf,
                Tree(child_tree) => {
                    let child = prune_rec(child_tree);
                    self.node_update(index, |_| child);
                }
            }
        }
    }

    /// Render octree as graphviz digraph.
    pub fn to_digraph(&self) -> String {
        let mut next = 1;
        let mut out = Vec::new();
        let mut run = || -> Result<String, Error> {
            writeln!(&mut out, "digraph OcTree {{")?;
            writeln!(&mut out, "  rankdir=\"LR\"")?;
            writeln!(
                &mut out,
                "  0 [label=\"{} {}\"]",
                self.info.leaf_count,
                self.info.min_color_count.unwrap_or(0),
            )?;
            self.to_digraph_rec(0, &mut next, &mut out)?;
            writeln!(&mut out, "}}")?;
            Ok(String::from_utf8_lossy(&out).into())
        };
        run().expect("memory write failed")
    }

    pub fn to_digraph_rec(
        &self,
        parent: usize,
        next: &mut usize,
        out: &mut Vec<u8>,
    ) -> Result<(), Error> {
        use OcTreeNode::*;
        for child in self.children.iter() {
            match child {
                Empty => continue,
                Leaf(leaf) => {
                    let id = *next;
                    *next += 1;

                    let fg = leaf
                        .to_rgba()
                        .best_contrast(RGBA::new(255, 255, 255, 255), RGBA::new(0, 0, 0, 255));
                    writeln!(
                        out,
                        "  {} [style=filled, fontcolor=\"{}\" fillcolor=\"{}\", label=\"{}\"]",
                        id,
                        fg,
                        leaf.to_rgba(),
                        leaf.color_count
                    )?;
                    writeln!(out, "  {} -> {}", parent, id)?;
                }
                Tree(tree) => {
                    let id = *next;
                    *next += 1;

                    writeln!(
                        out,
                        "  {} [label=\"{} {}\"]",
                        id,
                        tree.info.leaf_count,
                        tree.info.min_color_count.unwrap_or(0),
                    )?;
                    writeln!(out, "  {} -> {}", parent, id)?;
                    tree.to_digraph_rec(id, next, out)?
                }
            }
        }
        Ok(())
    }
}

/// Convert path generated by OcTreePath back to RGBA color
fn color_from_path(path: &Vec<usize>) -> RGBA {
    let mut r: u8 = 0;
    let mut g: u8 = 0;
    let mut b: u8 = 0;
    for index in 0..8 {
        r <<= 1;
        g <<= 1;
        b <<= 1;
        let bits = path.get(index).unwrap_or(&0);
        if bits & 0b100 != 0 {
            r |= 1;
        }
        if bits & 0b010 != 0 {
            g |= 1;
        }
        if bits & 0b001 != 0 {
            b |= 1;
        }
    }
    RGBA::new(r, g, b, 255)
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
    rgba: RGBA,
    state: u32,
    length: u8,
}

impl OcTreePath {
    pub fn new(rgba: RGBA) -> Self {
        let [r, g, b] = rgba.rgb_u8();
        // pack RGB components into u32 value
        let state = ((r as u32) << 16) | ((g as u32) << 8) | b as u32;
        Self {
            rgba,
            state,
            length: 8,
        }
    }

    pub fn rgba(&self) -> RGBA {
        self.rgba
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

pub struct ColorPalette {
    // there is no point in adding cache here as it always misses
    colors: Vec<RGBA>,
}

impl ColorPalette {
    pub fn new(colors: Vec<RGBA>) -> Option<Self> {
        if colors.is_empty() {
            None
        } else {
            Some(Self { colors })
        }
    }

    pub fn size(&self) -> usize {
        self.colors.len()
    }

    pub fn get(&self, index: usize) -> RGBA {
        self.colors[index]
    }

    pub fn colors(&self) -> &[RGBA] {
        &self.colors
    }

    pub fn find(&mut self, color: RGBA) -> (usize, RGBA) {
        fn dist(c0: RGBA, c1: RGBA) -> i32 {
            let [r0, g0, b0] = c0.rgb_u8();
            let [r1, g1, b1] = c1.rgb_u8();
            (r0 as i32 - r1 as i32).pow(2)
                + (g0 as i32 - g1 as i32).pow(2)
                + (b0 as i32 - b1 as i32).pow(2)
        }
        /*
        let (qindex, qcolor) = self
            .colors
            .iter()
            .enumerate()
            .min_by_key(|(_, c)| dist(**c, color))
            .unwrap();
        (qindex, *qcolor)
        */

        let best_dist = dist(color, self.colors[0]);
        let (best_index, _) =
            (1..self.colors.len()).fold((0, best_dist), |(best_index, best_dist), index| {
                let dist = dist(color, self.colors[index]);
                if dist < best_dist {
                    (index, dist)
                } else {
                    (best_index, best_dist)
                }
            });
        (best_index, self.colors[best_index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_octree_path() -> Result<(), Error> {
        let c0 = "#5a719d".parse::<RGBA>()?;
        let path: Vec<_> = OcTreePath::new(c0).collect();
        assert_eq!(path, vec![1, 6, 2, 7, 5, 1, 4, 3]);
        assert_eq!(c0, color_from_path(&path));

        let c1 = "#808080".parse::<RGBA>()?;
        let path: Vec<_> = OcTreePath::new(c1).collect();
        assert_eq!(path, vec![7, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(c1, color_from_path(&vec![7]));

        let c2 = "#d3869b".parse::<RGBA>()?;
        let path: Vec<_> = OcTreePath::new(c2).collect();
        assert_eq!(path, vec![7, 4, 0, 5, 1, 2, 7, 5]);
        assert_eq!(c2, color_from_path(&path));

        Ok(())
    }

    #[test]
    fn test_octree_info() {
        let mut tree = OcTree::new();
        tree.node_update(1, |_| {
            OcTreeNode::Leaf(OcTreeLeaf {
                red_acc: 1,
                green_acc: 2,
                blue_acc: 3,
                color_count: 4,
                index: 0,
            })
        });
        assert_eq!(
            tree.info,
            OcTreeInfo {
                leaf_count: 1,
                color_count: 4,
                min_color_count: Some(4),
            }
        );
    }

    #[test]
    fn test_octree() -> Result<(), Error> {
        let c0 = "#5a719d".parse::<RGBA>()?;
        let c1 = "#d3869b".parse::<RGBA>()?;

        let mut tree = OcTree::new();

        tree.insert(c0);
        tree.insert(c0);
        assert_eq!(
            tree.info(),
            OcTreeInfo {
                color_count: 2,
                leaf_count: 1,
                min_color_count: Some(2),
            }
        );
        assert_eq!(tree.find(c0), Some((0, c0)));

        tree.insert(c1);
        assert_eq!(
            tree.info(),
            OcTreeInfo {
                color_count: 3,
                leaf_count: 2,
                min_color_count: Some(1),
            }
        );
        assert_eq!(tree.find(c1), Some((0, c1)));

        Ok(())
    }
}
