use std::{
    any::Any,
    fmt::Debug,
    ops::{Deref, DerefMut},
};

use smallvec::SmallVec;

use crate::{surface::view_shape, Position, Size, SurfaceMutView, TerminalSurface};

pub type ViewLayoutStore = TreeStore<Layout>;
pub type ViewLayout<'a> = TreeView<'a, Layout>;
pub type ViewMutLayout<'a> = TreeMutView<'a, Layout>;

/// Layout of the [View] determines its position and size
#[derive(Default)]
pub struct Layout {
    pos: Position,
    size: Size,
    data: Option<Box<dyn Any>>,
}

impl std::cmp::PartialEq for Layout {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos
            && self.size == other.size
            && self.data.is_none()
            && other.data.is_none()
    }
}

impl std::cmp::Eq for Layout {}

impl Debug for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(if self.data.is_some() {
            "Layout"
        } else {
            "*Layout"
        })
        .field("row", &self.pos.row)
        .field("col", &self.pos.col)
        .field("height", &self.size.height)
        .field("width", &self.size.width)
        .finish()
    }
}

impl Layout {
    /// Create a new empty layout
    pub fn new() -> Self {
        Self::default()
    }

    /// Override layout position
    pub fn with_position(self, pos: Position) -> Self {
        Self { pos, ..self }
    }

    /// Get layout position
    pub fn pos(&self) -> Position {
        self.pos
    }

    /// Set layout position
    pub fn set_pos(&mut self, pos: Position) -> &mut Self {
        self.pos = pos;
        self
    }

    /// Override layout size
    pub fn with_size(self, size: Size) -> Self {
        Self { size, ..self }
    }

    /// Get layout size
    pub fn size(&self) -> Size {
        self.size
    }

    /// Set layout size
    pub fn set_size(&mut self, size: Size) -> &mut Self {
        self.size = size;
        self
    }

    /// Get layout data
    pub fn data<T: Any>(&self) -> Option<&T> {
        self.data.as_ref()?.downcast_ref()
    }

    /// Set layout data
    pub fn set_data(&mut self, data: impl Any) -> &mut Self {
        self.data = Some(Box::new(data));
        self
    }

    pub fn with_data(self, data: impl Any) -> Self {
        Self {
            data: Some(Box::new(data)),
            ..self
        }
    }

    /// Constrain surface by the layout, that is create sub-subsurface view
    /// with offset `pos` and size of `size`.
    pub fn apply_to<'a>(&self, surf: TerminalSurface<'a>) -> TerminalSurface<'a> {
        let rows = self.pos.row..self.pos.row + self.size.height;
        let cols = self.pos.col..self.pos.col + self.size.width;
        let (shape, data) = surf.parts();
        SurfaceMutView::new(view_shape(shape, rows, cols), data)
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct TreeId(usize);

pub type TreeStore<T> = SmallVec<[TreeNode<T>; 5]>;

impl Debug for TreeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TreeNode<T> {
    value: T,
    sibling: Option<TreeId>,
    child_first: Option<TreeId>,
    child_last: Option<TreeId>,
}

impl<T> TreeNode<T> {
    pub fn new(value: T) -> Self {
        Self {
            value,
            sibling: None,
            child_first: None,
            child_last: None,
        }
    }
}

pub trait Tree {
    type Value;

    /// Get tree id
    fn id(&self) -> TreeId;

    /// Node store
    fn store(&self) -> &[TreeNode<Self::Value>];

    /// Get view of this tree
    fn view(&self) -> TreeView<'_, Self::Value> {
        TreeView {
            store: self.store(),
            id: self.id(),
        }
    }

    /// Get immutable reference to the value associated with the root node
    fn value(&self) -> &Self::Value {
        &self.store()[self.id().0].value
    }

    /// Children iterator
    fn children(&self) -> TreeIter<'_, Self::Value> {
        TreeIter {
            store: self.store(),
            id: self.store()[self.id().0].child_first,
        }
    }

    /// Find path in the layout that leads to the position
    fn find_path(&self, pos: Position) -> FindPath<'_>
    where
        Self: Tree<Value = Layout>,
    {
        FindPath {
            current: Some(self.id()),
            store: self.store(),
            pos,
        }
    }
}

impl<'a, T: Tree> Tree for &'a T {
    type Value = T::Value;

    fn id(&self) -> TreeId {
        (**self).id()
    }

    fn store(&self) -> &[TreeNode<Self::Value>] {
        (**self).store()
    }
}

impl<'a, T: Tree> Tree for &'a mut T {
    type Value = T::Value;

    fn id(&self) -> TreeId {
        (**self).id()
    }

    fn store(&self) -> &[TreeNode<Self::Value>] {
        (**self).store()
    }
}

pub struct TreeIter<'a, T> {
    id: Option<TreeId>,
    store: &'a [TreeNode<T>],
}

impl<'a, T> Iterator for TreeIter<'a, T> {
    type Item = TreeView<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let id = self.id?;
        self.id = self.store[id.0].sibling;
        Some(TreeView {
            store: self.store,
            id,
        })
    }
}

pub struct FindPath<'a> {
    current: Option<TreeId>,
    store: &'a [TreeNode<Layout>],
    pos: Position,
}

impl<'a> Iterator for FindPath<'a> {
    type Item = &'a Layout;

    fn next(&mut self) -> Option<Self::Item> {
        let current_id = self.current.take()?;
        let mut child_id_opt = self.store[current_id.0].child_first;
        while let Some(child_id) = child_id_opt {
            let child = &self.store[child_id.0].value;
            if child.pos.col <= self.pos.col
                && self.pos.col < child.pos.col + child.size.width
                && child.pos.row <= self.pos.row
                && self.pos.row < child.pos.row + child.size.height
            {
                self.pos = Position {
                    row: self.pos.row - child.pos.row,
                    col: self.pos.col - child.pos.col,
                };
                self.current.replace(child_id);
                break;
            }
            child_id_opt = self.store[child_id.0].sibling;
        }
        Some(&self.store[current_id.0].value)
    }
}

pub trait TreeMut: Tree {
    /// Get mutable reference to the vector of node store
    fn store_mut(&mut self) -> &mut TreeStore<Self::Value>;

    /// Get mutable view of this tree
    fn view_mut(&mut self) -> TreeMutView<'_, Self::Value> {
        let id = self.id();
        TreeMutView {
            store: self.store_mut(),
            id,
        }
    }

    /// Get mutable reference to the value associated with the root node
    fn value_mut(&mut self) -> &mut Self::Value {
        let id = self.id().0;
        &mut self.store_mut()[id].value
    }

    /// Add new child with the provided value, returns child tree view
    fn push(&mut self, value: Self::Value) -> TreeMutView<'_, Self::Value> {
        let root_id = self.id().0;
        let store = self.store_mut();

        // allocate node
        let child_id = TreeId(store.len());
        store.push(TreeNode::new(value));

        // update root and sibling
        match store[root_id].child_last.replace(child_id) {
            None => {
                store[root_id].child_first.replace(child_id);
            }
            Some(sibling_id) => {
                store[sibling_id.0].sibling.replace(child_id);
            }
        }

        TreeMutView {
            store,
            id: child_id,
        }
    }

    /// Add new child with default value, returns child tree view
    fn push_default(&mut self) -> TreeMutView<'_, Self::Value>
    where
        Self::Value: Default,
    {
        self.push(Default::default())
    }

    /// Pop first child, returns value associated with the child
    fn pop(&mut self) -> Option<TreeMutView<'_, Self::Value>> {
        let root_id = self.id().0;
        let store = self.store_mut();

        let child_id = store[root_id].child_first.take()?;
        let sibling_id = store[child_id.0].sibling.take();
        store[root_id].child_first = sibling_id;
        if sibling_id.is_none() {
            store[root_id].child_last = None;
        }

        Some(TreeMutView {
            store,
            id: child_id,
        })
    }

    /// Get mutable tree view of the first child
    fn child_mut(&mut self) -> Option<TreeMutView<'_, Self::Value>> {
        let child_id = self.store()[self.id().0].child_first?;
        Some(TreeMutView {
            store: self.store_mut(),
            id: child_id,
        })
    }
}

impl<'a, T: TreeMut> TreeMut for &'a mut T {
    fn store_mut(&mut self) -> &mut TreeStore<Self::Value> {
        (**self).store_mut()
    }
}

pub struct TreeView<'a, T> {
    store: &'a [TreeNode<T>],
    id: TreeId,
}

impl<'a, T> TreeView<'a, T> {
    pub fn new(store: &'a mut TreeStore<T>, value: T) -> Self {
        let id = TreeId(store.len());
        store.push(TreeNode::new(value));
        Self {
            store: store.as_slice(),
            id,
        }
    }

    pub fn from_id(store: &'a TreeStore<T>, id: TreeId) -> Self {
        Self {
            store: store.as_slice(),
            id,
        }
    }
}

impl<'a, T> Deref for TreeView<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value()
    }
}

impl<'a, T> Tree for TreeView<'a, T> {
    type Value = T;

    fn id(&self) -> TreeId {
        self.id
    }

    fn store(&self) -> &[TreeNode<Self::Value>] {
        self.store
    }
}

impl<'a, T: Debug> Debug for TreeView<'a, T> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn debug_rec<T: Debug>(
            this: TreeView<'_, T>,
            offset: usize,
            fmt: &mut std::fmt::Formatter<'_>,
        ) -> std::fmt::Result {
            writeln!(fmt, "{0:<1$}{2:?}", "", offset, this.value())?;
            for child in this.children() {
                debug_rec(child, offset + 2, fmt)?;
            }
            Ok(())
        }
        writeln!(fmt)?;
        debug_rec(self.view(), 0, fmt)?;
        Ok(())
    }
}

impl<'a, T, O> PartialEq<O> for TreeView<'a, T>
where
    T: PartialEq,
    O: Tree<Value = T>,
{
    fn eq(&self, other: &O) -> bool {
        if self.value() != other.value() {
            return false;
        }
        let mut left_iter = self.children();
        let mut right_iter = other.children();
        loop {
            match (left_iter.next(), right_iter.next()) {
                (None, None) => return true,
                (Some(left), Some(right)) => {
                    if left != right {
                        return false;
                    } else {
                        continue;
                    }
                }
                _ => return false,
            }
        }
    }
}

impl<'a, T: Eq> Eq for TreeView<'a, T> {}

pub struct TreeMutView<'a, T> {
    store: &'a mut TreeStore<T>,
    id: TreeId,
}

impl<'a, T> TreeMutView<'a, T> {
    pub fn new(store: &'a mut TreeStore<T>, value: T) -> Self {
        let id = TreeId(store.len());
        store.push(TreeNode::new(value));
        Self { store, id }
    }

    pub fn from_id(store: &'a mut TreeStore<T>, id: TreeId) -> Self {
        Self { store, id }
    }

    pub fn sibling(mut self) -> Option<TreeMutView<'a, T>> {
        self.id = self.store()[self.id().0].sibling?;
        Some(self)
    }
}

impl<'a, T> Deref for TreeMutView<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value()
    }
}

impl<'a, T> DerefMut for TreeMutView<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value_mut()
    }
}

impl<'a, T> Tree for TreeMutView<'a, T> {
    type Value = T;

    fn id(&self) -> TreeId {
        self.id
    }

    fn store(&self) -> &[TreeNode<Self::Value>] {
        self.store
    }
}

impl<'a, T> TreeMut for TreeMutView<'a, T> {
    fn store_mut(&mut self) -> &mut TreeStore<T> {
        self.store
    }
}

impl<'a, T: Debug> Debug for TreeMutView<'a, T> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.view().fmt(fmt)
    }
}

impl<'a, T, O> PartialEq<O> for TreeMutView<'a, T>
where
    T: PartialEq,
    O: Tree<Value = T>,
{
    fn eq(&self, other: &O) -> bool {
        self.view() == other
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_tree() -> Result<(), Box<dyn Error>> {
        let mut s0 = TreeStore::new();
        let mut t0 = TreeMutView::new(&mut s0, "a");
        t0.push("b");
        t0.push("c").push("d");

        let mut s1 = TreeStore::new();
        let mut t1 = TreeMutView::new(&mut s1, "a");
        t1.push("b");
        t1.push("c").push("d");

        assert_eq!(t0, t1.view());

        t1.push("e");
        assert_ne!(t0, t1);
        assert_eq!(
            t1.children().map(|v| *v).collect::<Vec<_>>(),
            vec!["b", "c", "e"]
        );

        assert_eq!(*t0.pop().expect("first child - c"), "b");
        assert_eq!(*t0.pop().expect("first child - c"), "c");
        assert!(t0.pop().is_none());
        let n0 = t0.store()[t0.id().0];
        assert!(n0.child_first.is_none());
        assert!(n0.child_last.is_none());

        Ok(())
    }

    #[test]
    fn test_layout_find_path() {
        let mut layout_store = ViewLayoutStore::new();
        let mut layout = ViewMutLayout::new(
            &mut layout_store,
            Layout::new().with_size(Size::new(10, 10)),
        );
        layout.push(Layout::new().with_size(Size::new(6, 6)));
        layout.push(
            Layout::new()
                .with_position(Position::new(6, 0))
                .with_size(Size::new(4, 5)),
        );
        layout.push(
            Layout::new()
                .with_position(Position::new(6, 5))
                .with_size(Size::new(4, 5)),
        );
        let mut layout1 = layout.push(
            Layout::new()
                .with_position(Position::new(0, 6))
                .with_size(Size::new(6, 4)),
        );
        layout1.push(Layout::new().with_size(Size::new(3, 4)));
        layout1.push(
            Layout::new()
                .with_position(Position::new(3, 0))
                .with_size(Size::new(3, 4)),
        );

        assert_eq!(
            layout.find_path(Position::new(4, 7)).collect::<Vec<_>>(),
            vec![
                &Layout::new().with_size(Size::new(10, 10)),
                &Layout::new()
                    .with_position(Position::new(0, 6))
                    .with_size(Size::new(6, 4)),
                &Layout::new()
                    .with_position(Position::new(3, 0))
                    .with_size(Size::new(3, 4)),
            ]
        );
    }
}
