use crate::{
    encoder::base64_encode, Color, ColorLinear, Error, Position, Shape, Surface, Terminal,
    TerminalEvent, RGBA,
};
use std::{
    collections::HashMap,
    fmt,
    fmt::Write as _,
    io::{Cursor, Read, Write},
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

#[derive(Debug, Clone, Copy)]
struct OcTreeNodeStats {
    leaf_count: usize,
    min_color: Option<usize>,
}

impl OcTreeNodeStats {
    fn empty() -> Self {
        Self {
            leaf_count: 0,
            min_color: None,
        }
    }

    fn join(self, other: Self) -> Self {
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

    fn leaf_count(&self) -> usize {
        use OcTreeNode::*;
        match self {
            Empty => 0,
            Leaf(_) => 1,
            Tree(tree) => tree.leaf_count,
        }
    }

    fn stats(&self) -> OcTreeNodeStats {
        use OcTreeNode::*;
        match self {
            Empty => OcTreeNodeStats::empty(),
            Leaf(leaf) => OcTreeNodeStats {
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
    stats: OcTreeNodeStats,
    color_count: usize,
    leaf_count: usize,
    children: [OcTreeNode; 8],
}

impl OcTree {
    pub fn new() -> Self {
        use OcTreeNode::Empty;
        Self {
            stats: OcTreeNodeStats::empty(),
            color_count: 0,
            leaf_count: 0,
            children: [Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty],
        }
    }

    pub fn color_count(&self) -> usize {
        self.color_count
    }

    pub fn leaf_count(&self) -> usize {
        self.leaf_count
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

    /// Repalce node and fix node statistics
    fn node_replace(&mut self, index: usize, node: OcTreeNode) -> OcTreeNode {
        let old_node = std::mem::replace(&mut self.children[index], node);
        self.stats = OcTreeNodeStats::from_slice(&self.children);
        old_node
    }

    /// Take node (Note: statistics is broken at this point)
    fn node_take(&mut self, index: usize) -> OcTreeNode {
        self.children[index].take()
    }

    /// Insert color into the octree
    pub fn insert(&mut self, color: impl Color) {
        let mut path = OcTreePath::new(color.rgb_u8());
        let index = path.next().expect("OcTreePath can not be empty");
        let (child, leafs) = Self::insert_rec(self.node_take(index), path);
        self.children[index] = child;
        self.color_count += 1;
        self.leaf_count += leafs;
    }

    /// Recursive insertion of the color into a node
    ///
    /// Returns updated node and number of leafs added.
    fn insert_rec(mut node: OcTreeNode, mut path: OcTreePath) -> (OcTreeNode, usize) {
        use OcTreeNode::*;
        match path.next() {
            Some(index) => match node {
                Empty => {
                    let (child, leafs) = Self::insert_rec(Empty, path);
                    let mut tree = Self::new();
                    tree.children[index] = child;
                    tree.color_count += 1;
                    tree.leaf_count += leafs;
                    (Tree(Box::new(tree)), leafs)
                }
                Leaf(mut leaf) => {
                    let [r, g, b] = path.rgb();
                    leaf.red_acc += r as usize;
                    leaf.green_acc += g as usize;
                    leaf.blue_acc += b as usize;
                    leaf.color_count += 1;
                    (node, 0)
                }
                Tree(ref mut tree) => {
                    let (child, leafs) = Self::insert_rec(tree.children[index].take(), path);
                    tree.children[index] = child;
                    tree.color_count += 1;
                    tree.leaf_count += leafs;
                    (node, leafs)
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
                    (Leaf(leaf), 1)
                }
                Leaf(mut leaf) => {
                    let [r, g, b] = path.rgb();
                    leaf.red_acc += r as usize;
                    leaf.green_acc += g as usize;
                    leaf.blue_acc += b as usize;
                    leaf.color_count += 1;
                    (node, 0)
                }
                Tree(_) => unreachable!(),
            },
        }
    }

    /// Prune octree
    ///
    /// Search the node, where the sum of the childs references is minimal and
    /// convert it to a leaf.
    pub fn prune(&mut self) {
        if let Some(index) = self.find_argmin_count_tree() {
            let child = self.children[index].take();
            self.leaf_count -= child.leaf_count();
            let new_child = Self::prune_rec(child);
            self.leaf_count += new_child.leaf_count();
            self.children[index] = new_child;
        }
    }

    /// Recursive pruning of the sub-tree
    fn prune_rec(node: OcTreeNode) -> OcTreeNode {
        use OcTreeNode::*;
        let mut tree = match node {
            Tree(tree) => tree,
            _ => return node,
        };
        match tree.find_argmin_count_tree() {
            None => {
                // we have found the node with minimal number of refernces
                let mut leaf = OcTreeLeaf::new();
                for index in 0..8 {
                    match tree.children[index].take() {
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
                let child = tree.children[index].take();
                tree.leaf_count -= child.leaf_count();
                let new_child = Self::prune_rec(child);
                tree.leaf_count += new_child.leaf_count();
                tree.children[index] = new_child;
                Tree(tree)
            }
        }
    }

    /// Find **sub-tree** with minimum count of colors
    fn find_argmin_count_tree(&self) -> Option<usize> {
        let mut min = None;
        let mut argmin = None;
        for index in 0..8 {
            let color_count = match &self.children[index] {
                OcTreeNode::Tree(tree) => tree.color_count,
                _ => continue,
            };
            match min {
                None => {
                    min = Some(color_count);
                    argmin = Some(index);
                }
                Some(min_count) if color_count < min_count => {
                    min = Some(color_count);
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
            "  0 [label=\"{} {}\"]",
            self.leaf_count, self.color_count
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

                    writeln!(
                        out,
                        "  {} [label=\"{} [{}]\"]",
                        id, tree.color_count, tree.leaf_count
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

pub struct ColorPaletteIndex(usize);

pub struct ColorPalette {
    colors: Vec<ColorLinear>,
}

impl ColorPalette {
    pub fn new<C, CS>(colors: CS) -> Option<Self>
    where
        CS: IntoIterator<Item = C>,
        C: Color,
    {
        // TODO:
        //   - Account for alpha by blending with background
        //   - Acutally compute palette
        let colors: Vec<ColorLinear> = colors.into_iter().map(Into::into).collect();
        if colors.is_empty() {
            Some(Self { colors })
        } else {
            None
        }
    }

    pub fn find<C: Color>(&self, color: C) -> ColorPaletteIndex {
        let color: ColorLinear = color.into();
        let best_dist = self.colors[0].distance(&color);
        let (best_index, _dist) =
            (1..self.colors.len()).fold((0, best_dist), |(best_index, best_dist), index| {
                let dist = self.colors[index].distance(&color);
                if dist < best_dist {
                    (index, dist)
                } else {
                    (best_index, best_dist)
                }
            });
        ColorPaletteIndex(best_index)
    }

    pub fn get(&self, index: ColorPaletteIndex) -> ColorLinear {
        self.colors[index.0]
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
    fn test_octree() -> Result<(), Error> {
        let c0 = "#5a719d".parse::<RGBA>()?;
        let c1 = "#d3869b".parse::<RGBA>()?;

        let mut tree = OcTree::new();

        tree.insert(c0);
        tree.insert(c0);
        assert_eq!(tree.color_count, 2);
        assert_eq!(tree.leaf_count, 1);
        assert_eq!(tree.find_argmin_count_tree(), Some(1));
        assert_eq!(tree.find(c0), Some(c0));

        tree.insert(c1);
        assert_eq!(tree.color_count, 3);
        assert_eq!(tree.leaf_count, 2);
        assert_eq!(tree.find_argmin_count_tree(), Some(7));
        assert_eq!(tree.find(c1), Some(c1));

        Ok(())
    }
}
