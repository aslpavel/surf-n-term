use crate::{
    encoder::base64_encode, Color, ColorLinear, Error, Position, Shape, Surface, Terminal,
    TerminalEvent, RGBA,
};
use std::{
    collections::HashMap,
    fmt,
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
    Ok(Some(Box::new(KittyImageHandler::new())))
}

/// Image handler for kitty graphic protocol
///
/// Reference: https://sw.kovidgoyal.net/kitty/graphics-protocol.html
pub struct KittyImageHandler {
    imgs: HashMap<u64, usize>, // hash -> size in bytes
}

impl KittyImageHandler {
    pub fn new() -> Self {
        Self {
            imgs: Default::default(),
        }
    }
}

impl Default for KittyImageHandler {
    fn default() -> Self {
        Self::new()
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
                        "\x1b_Ga=T,f=32,o=z,v={},s={},m={},i={};",
                        img.height(),
                        img.width(),
                        more,
                        id,
                    )?;
                    out.write_all(&buf[..size])?;
                    out.write_all(b"\x1b\\")?;
                    if more == 0 {
                        break;
                    }
                }
                self.imgs.insert(id, img.size());
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

#[derive(Debug)]
struct OcTreeLeaf {
    red_acc: usize,
    green_acc: usize,
    blue_acc: usize,
    count: usize,
}

#[derive(Debug)]
enum OcTreeNode {
    Leaf(OcTreeLeaf),
    Tree(OcTree),
}

#[derive(Debug)]
pub struct OcTree {
    // number of colors in the sub-tree
    count: usize,
    // number of leafs in the sub-tree
    leafs: usize,
    // children nodes
    children: [Option<Box<OcTreeNode>>; 8],
}

impl OcTree {
    pub fn new() -> Self {
        Self {
            count: 0,
            leafs: 0,
            children: [None, None, None, None, None, None, None, None],
        }
    }

    /// Find nearest color inside the octree
    pub fn find(&self, color: impl Color) -> Option<RGBA> {
        let mut tree = self;
        for index in OcTreePath::new(color.rgb_u8()) {
            match tree.children[index].as_ref()?.as_ref() {
                OcTreeNode::Leaf(leaf) => {
                    let r = (leaf.red_acc / leaf.count) as u8;
                    let g = (leaf.green_acc / leaf.count) as u8;
                    let b = (leaf.blue_acc / leaf.count) as u8;
                    return Some(RGBA::new(r, g, b, 255));
                }
                OcTreeNode::Tree(next_tree) => tree = next_tree,
            }
        }
        None
    }

    /// Insert color into the octree
    pub fn insert(&mut self, color: impl Color) {
        let mut path = OcTreePath::new(color.rgb_u8());
        let index = path.next().expect("OcTreePath can not be empty");
        let (child, leafs) = Self::insert_rec(self.children[index].take(), path);
        self.children[index] = child;
        self.count += 1;
        self.leafs += leafs;
    }

    /// Recursive insertion of the color into a node
    ///
    /// Returns updated node and number of leafs added.
    fn insert_rec(
        node: Option<Box<OcTreeNode>>,
        mut path: OcTreePath,
    ) -> (Option<Box<OcTreeNode>>, usize) {
        match path.next() {
            Some(index) => match node {
                Some(mut node) => match node.as_mut() {
                    OcTreeNode::Leaf(leaf) => {
                        let [r, g, b] = path.rgb();
                        leaf.red_acc += r as usize;
                        leaf.green_acc += g as usize;
                        leaf.blue_acc += b as usize;
                        leaf.count += 1;
                        (Some(node), 0)
                    }
                    OcTreeNode::Tree(tree) => {
                        let (child, leafs) = Self::insert_rec(tree.children[index].take(), path);
                        tree.children[index] = child;
                        tree.count += 1;
                        tree.leafs += leafs;
                        (Some(node), leafs)
                    }
                },
                None => {
                    let (child, leafs) = Self::insert_rec(None, path);
                    let mut tree = Self::new();
                    tree.children[index] = child;
                    tree.count += 1;
                    tree.leafs += leafs;
                    (Some(Box::new(OcTreeNode::Tree(tree))), leafs)
                }
            },
            None => match node {
                Some(mut node) => match node.as_mut() {
                    OcTreeNode::Leaf(leaf) => {
                        let [r, g, b] = path.rgb();
                        leaf.red_acc += r as usize;
                        leaf.green_acc += g as usize;
                        leaf.blue_acc += b as usize;
                        leaf.count += 1;
                        (Some(node), 0)
                    }
                    _ => unreachable!(),
                },
                None => {
                    let [r, g, b] = path.rgb();
                    let leaf = OcTreeLeaf {
                        red_acc: r as usize,
                        green_acc: g as usize,
                        blue_acc: b as usize,
                        count: 1,
                    };
                    (Some(Box::new(OcTreeNode::Leaf(leaf))), 1)
                }
            },
        }
    }

    /*
    /// Prune octree
    ///
    /// Search the node, where the sum of the childs references is minimal and
    /// convert it to a leaf.
    pub fn prune(&mut self) {
        let index = match self.find_count_argmin() {
            None => return,
            Some(index) => index,
        };
    }

    /// Recursive pruning of the sub-tree, returns number of leafs in
    fn prune_rec(&mut self) -> usize {
        let index = match self.find_count_argmin() {
            None => return,
            Some(index) => index,
        };
        match self.children[index] {
            None => unreachable!("find_count_argmin returned empty subtree index"),
            Some(node) => match node.as_mut() {
                OcTreeNode::Leaf(_) => {}
            },
        }
    }
    */

    /// Find sub-tree index with minimum count of colors
    fn find_count_argmin(&self) -> Option<usize> {
        let mut min = None;
        let mut argmin = None;
        for index in 0..8 {
            let count = match &self.children[index] {
                None => continue,
                Some(node) => match node.as_ref() {
                    OcTreeNode::Leaf(leaf) => leaf.count,
                    OcTreeNode::Tree(tree) => tree.count,
                },
            };
            match min {
                None => {
                    min = Some(count);
                    argmin = Some(index);
                }
                Some(min_count) if count < min_count => {
                    min = Some(count);
                    argmin = Some(index);
                }
                _ => (),
            }
        }
        argmin
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
        assert_eq!(tree.count, 2);
        assert_eq!(tree.leafs, 1);
        assert_eq!(tree.find_count_argmin(), Some(1));
        assert_eq!(tree.find(c0), Some(c0));

        tree.insert(c1);
        assert_eq!(tree.count, 3);
        assert_eq!(tree.leafs, 2);
        assert_eq!(tree.find_count_argmin(), Some(7));
        assert_eq!(tree.find(c1), Some(c1));

        Ok(())
    }
}
