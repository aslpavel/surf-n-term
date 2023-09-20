//! Surface object
//!
//! Matrix like object which is used to access and modify terminal frame and images
#![allow(clippy::iter_nth_zero)]

use std::{
    hash::{Hash, Hasher},
    ops::{
        Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
    },
    sync::Arc,
};

use crate::{common::clamp, Position, Size};

/// Shape object describing layout of data in the surface object
#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct Shape {
    /// Offset of the first element.
    pub start: usize,
    /// Offset of the last + 1 element.
    pub end: usize,
    /// Width of the surface
    pub width: usize,
    /// Height of the surface
    pub height: usize,
    /// How many elements we need to skip to get to the next row.
    pub row_stride: usize,
    /// How many elements we need to skip to get to the next column.
    pub col_stride: usize,
}

impl Shape {
    /// Convert row and col to offset.
    #[inline]
    pub fn offset(&self, pos: Position) -> usize {
        self.start + pos.row * self.row_stride + pos.col * self.col_stride
    }

    /// Get current index of the in row-major order corresponding
    /// to provided row and column.
    #[inline]
    pub fn index(&self, pos: Position) -> usize {
        pos.row * self.width + pos.col
    }

    /// Get row and column corresponding to nth element in row-major order
    #[inline]
    pub fn nth(&self, n: usize) -> Option<Position> {
        if self.width == 0 {
            return None;
        }
        let row = n / self.width;
        let col = n - row * self.width;
        (row < self.height).then_some(Position { row, col })
    }

    /// Get surface size
    #[inline]
    pub fn size(&self) -> Size {
        Size {
            width: self.width,
            height: self.height,
        }
    }
}

impl From<Size> for Shape {
    fn from(size: Size) -> Self {
        Self {
            start: 0,
            end: size.height * size.width,
            width: size.width,
            height: size.height,
            row_stride: size.width,
            col_stride: 1,
        }
    }
}

/// Matrix like object used to store `RGBA` (in case of images) and `Cell` (in case of Terminal)
pub trait Surface {
    type Item;

    /// Shape describes data layout inside `Self::data()` slice.
    fn shape(&self) -> Shape;

    /// Slice containing all the items
    ///
    /// **Note:** This slice contains all elements backed by parent object
    /// and elements should be accessed using the offset calculated by `Shape::offset`
    fn data(&self) -> &[Self::Item];

    /// Check if surface is empty
    fn is_empty(&self) -> bool {
        let shape = self.shape();
        shape.start >= shape.end
    }

    /// Height of the surface
    fn height(&self) -> usize {
        self.shape().height
    }

    /// Width of the surface
    fn width(&self) -> usize {
        self.shape().width
    }

    /// Size of the surface
    fn size(&self) -> Size {
        self.shape().size()
    }

    fn hash(&self) -> u64
    where
        Self::Item: Hash,
    {
        let mut hasher = fnv::FnvHasher::default();
        hasher.write_usize(self.height());
        hasher.write_usize(self.width());
        for item in self.iter() {
            item.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Get immutable reference to the element specified by row and column
    fn get(&self, pos: Position) -> Option<&Self::Item> {
        let shape = self.shape();
        if pos.row >= shape.height || pos.col >= shape.width {
            None
        } else {
            self.data().get(shape.offset(pos))
        }
    }

    /// Iterator over immutable references to the items of the view in the row-major order
    fn iter(&self) -> SurfaceIter<'_, Self::Item> {
        SurfaceIter {
            index: 0,
            shape: self.shape(),
            data: self.data(),
        }
    }

    /// Create an immutable sub-surface restricted by `rows` and `cols` bounds.
    fn view<RS, CS>(&self, rows: RS, cols: CS) -> SurfaceView<'_, Self::Item>
    where
        RS: ViewBounds,
        CS: ViewBounds,
        Self: Sized,
    {
        SurfaceView {
            shape: view_shape(self.shape(), rows, cols),
            data: self.data(),
        }
    }

    fn as_ref(&self) -> SurfaceView<'_, Self::Item> {
        SurfaceView {
            shape: self.shape(),
            data: self.data(),
        }
    }

    /// Create owned sub-surface restricted by `rows` and `cols` bounds.
    fn view_owned<RS, CS>(self, rows: RS, cols: CS) -> SurfaceOwnedView<Self>
    where
        RS: ViewBounds,
        CS: ViewBounds,
        Self: Sized,
    {
        SurfaceOwnedView {
            shape: view_shape(self.shape(), rows, cols),
            inner: self,
        }
    }

    // Crate transposed (same way as matrix is transposed) view of the surface.
    fn transpose(self) -> SurfaceOwnedView<Self>
    where
        Self: Sized,
    {
        let shape = self.shape();
        let shape = Shape {
            width: shape.height,
            height: shape.width,
            col_stride: shape.row_stride,
            row_stride: shape.col_stride,
            ..shape
        };
        SurfaceOwnedView { shape, inner: self }
    }

    // Create new owned surface (allocates) by mapping all elements with the function.
    fn map<F, T>(&self, mut f: F) -> SurfaceOwned<T>
    where
        F: FnMut(Position, &Self::Item) -> T,
        Self: Sized,
    {
        let shape = self.shape();
        let data = self.data();
        SurfaceOwned::new_with(shape.size(), |pos| f(pos, &data[shape.offset(pos)]))
    }

    /// Create owned copy of the surface
    fn to_owned_surf(&self) -> SurfaceOwned<Self::Item>
    where
        Self::Item: Clone,
    {
        let shape = self.shape();
        let data = self.data();
        SurfaceOwned::new_with(shape.size(), |pos| data[shape.offset(pos)].clone())
    }
}

/// Mutable surface
pub trait SurfaceMut: Surface {
    /// Mutable slice containing all the items
    fn data_mut(&mut self) -> &mut [Self::Item];

    /// Get mutable reference to the element specified by row and column
    fn get_mut(&mut self, pos: Position) -> Option<&mut Self::Item> {
        let shape = self.shape();
        if pos.row >= shape.height || pos.col >= shape.width {
            None
        } else {
            self.data_mut().get_mut(shape.offset(pos))
        }
    }

    /// Set value at row and column
    fn set(&mut self, pos: Position, item: Self::Item) -> Self::Item {
        let shape = self.shape();
        debug_assert!(
            pos.row < shape.height,
            "row {} is out of bound (height {})",
            pos.row,
            shape.height
        );
        debug_assert!(
            pos.col < shape.width,
            "column {} is out of bound (width {})",
            pos.col,
            shape.width
        );
        std::mem::replace(&mut self.data_mut()[shape.offset(pos)], item)
    }

    /// Iterator over mutable references to the items of the view in the row-major order
    fn iter_mut(&mut self) -> SurfaceMutIter<'_, Self::Item> {
        SurfaceMutIter {
            index: 0,
            shape: self.shape(),
            data: self.data_mut(),
        }
    }

    /// Create a mutable sub-surface restricted by `rows` and `cols` bounds.
    fn view_mut<RS, CS>(&mut self, rows: RS, cols: CS) -> SurfaceMutView<'_, Self::Item>
    where
        RS: ViewBounds,
        CS: ViewBounds,
        Self: Sized,
    {
        SurfaceMutView {
            shape: view_shape(self.shape(), rows, cols),
            data: self.data_mut(),
        }
    }

    fn as_mut(&mut self) -> SurfaceMutView<'_, Self::Item> {
        SurfaceMutView {
            shape: self.shape(),
            data: self.data_mut(),
        }
    }

    /// Fill all elements of the surface with the copy of provided item.
    fn fill(&mut self, item: Self::Item)
    where
        Self::Item: Clone,
    {
        let shape = self.shape();
        let data = self.data_mut();
        for row in 0..shape.height {
            for col in 0..shape.width {
                data[shape.offset(Position::new(row, col))] = item.clone();
            }
        }
    }

    /// Fill all the elements of the surface by calling a function.
    ///
    /// Function is called it row, column and the current item value as its arguments.
    fn fill_with<F>(&mut self, mut fill: F)
    where
        F: FnMut(Position, Self::Item) -> Self::Item,
        Self::Item: Default,
        Self: Sized,
    {
        let shape = self.shape();
        let data = self.data_mut();
        let mut tmp = Self::Item::default();
        for row in 0..shape.height {
            for col in 0..shape.width {
                let pos = Position::new(row, col);
                let offset = shape.offset(pos);
                let item = std::mem::replace(&mut data[offset], tmp);
                tmp = std::mem::replace(&mut data[offset], fill(pos, item));
            }
        }
    }

    /// Fill all the element of the surface with default value.
    fn clear(&mut self)
    where
        Self::Item: Default,
    {
        let shape = self.shape();
        let data = self.data_mut();
        for row in 0..shape.height {
            for col in 0..shape.width {
                data[shape.offset(Position::new(row, col))] = Default::default();
            }
        }
    }

    /// Insert items starting with the specified row and column.
    fn insert<IS>(&mut self, pos: Position, items: IS)
    where
        IS: IntoIterator<Item = Self::Item>,
        Self: Sized,
    {
        let index = pos.row * self.width() + pos.col;
        let mut iter = self.iter_mut();
        if index > 0 {
            iter.nth(index - 1);
        }
        for (src, dst) in items.into_iter().zip(iter) {
            *dst = src
        }
    }
}

/// Iterator over elements of the Surface
pub struct SurfaceIter<'a, T> {
    index: usize,
    shape: Shape,
    data: &'a [T],
}

impl<'a, T> SurfaceIter<'a, T> {
    /// Position of the element that will be yielded next
    pub fn position(&self) -> Position {
        self.shape
            .nth(self.index)
            .unwrap_or_else(|| Position::new(self.shape.height, 0))
    }

    /// Augment iterator with position
    pub fn with_position(self) -> SurfacePosIter<'a, T> {
        SurfacePosIter { iter: self }
    }

    /// Index of the element that will be yielded next
    pub fn index(&self) -> usize {
        self.index
    }

    /// Shape of the surface
    pub fn shape(&self) -> Shape {
        self.shape
    }
}

impl<'a, T: 'a> Iterator for SurfaceIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.nth(0)
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.index += n + 1;
        let pos = self.shape.nth(self.index - 1)?;
        self.data.get(self.shape.offset(pos))
    }
}

pub struct SurfacePosIter<'a, T> {
    iter: SurfaceIter<'a, T>,
}

impl<'a, T> Iterator for SurfacePosIter<'a, T> {
    type Item = (Position, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let pos = self.iter.position();
        Some((pos, self.iter.next()?))
    }
}

/// Iterator over mutable elements of the Surface
pub struct SurfaceMutIter<'a, T> {
    index: usize,
    shape: Shape,
    data: &'a mut [T],
}

impl<'a, T> SurfaceMutIter<'a, T> {
    /// Position of the element that will be yielded next
    pub fn position(&self) -> Position {
        self.shape
            .nth(self.index)
            .unwrap_or_else(|| Position::new(self.shape.height, 0))
    }

    /// Augment iterator with position
    pub fn with_position(self) -> SurfacePosMutIter<'a, T> {
        SurfacePosMutIter { iter: self }
    }

    /// Index of the element that will be yielded next
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }
}

impl<'a, T: 'a> Iterator for SurfaceMutIter<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        self.nth(0)
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.index += n + 1;
        let pos = self.shape.nth(self.index - 1)?;
        let offset = self.shape.offset(pos);

        if offset >= self.data.len() {
            None
        } else {
            // this is safe, iterator is always progressing and never
            // returns a mutable reference to the same location.
            let ptr = self.data.as_mut_ptr();
            let item = unsafe { &mut *ptr.add(offset) };
            Some(item)
        }
    }
}

pub struct SurfacePosMutIter<'a, T> {
    iter: SurfaceMutIter<'a, T>,
}

impl<'a, T> Iterator for SurfacePosMutIter<'a, T> {
    type Item = (Position, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        let pos = self.iter.position();
        Some((pos, self.iter.next()?))
    }
}

impl<'a, S> Surface for &'a S
where
    S: Surface + ?Sized,
{
    type Item = S::Item;

    fn shape(&self) -> Shape {
        (**self).shape()
    }

    fn data(&self) -> &[Self::Item] {
        (**self).data()
    }
}

impl<'a, S> Surface for &'a mut S
where
    S: Surface + ?Sized,
{
    type Item = S::Item;

    fn shape(&self) -> Shape {
        (**self).shape()
    }

    fn data(&self) -> &[Self::Item] {
        (**self).data()
    }
}

impl<'a, S> SurfaceMut for &'a mut S
where
    S: SurfaceMut + ?Sized,
{
    fn data_mut(&mut self) -> &mut [Self::Item] {
        (**self).data_mut()
    }
}

impl<S> Surface for Arc<S>
where
    S: Surface + ?Sized,
{
    type Item = S::Item;

    fn shape(&self) -> Shape {
        (**self).shape()
    }

    fn data(&self) -> &[Self::Item] {
        (**self).data()
    }
}

impl<S> Surface for Box<S>
where
    S: Surface + ?Sized,
{
    type Item = S::Item;

    fn shape(&self) -> Shape {
        (**self).shape()
    }

    fn data(&self) -> &[Self::Item] {
        (**self).data()
    }
}

impl<S> SurfaceMut for Box<S>
where
    S: SurfaceMut + ?Sized,
{
    fn data_mut(&mut self) -> &mut [Self::Item] {
        (**self).data_mut()
    }
}

/// Surface owns its data
#[derive(Clone)]
pub struct SurfaceOwned<T> {
    shape: Shape,
    data: Vec<T>,
}

/// View over owned surface
#[derive(Clone)]
pub struct SurfaceOwnedView<S> {
    shape: Shape,
    inner: S,
}

/// View of (sub)surface
#[derive(Clone)]
pub struct SurfaceView<'a, T> {
    shape: Shape,
    data: &'a [T],
}

/// Mutable view of (sub)surface
pub struct SurfaceMutView<'a, T> {
    shape: Shape,
    data: &'a mut [T],
}

impl<T> SurfaceOwned<T> {
    pub fn new(size: Size) -> Self
    where
        T: Default,
    {
        Self::new_with(size, |_| Default::default())
    }

    pub fn new_with<F>(size: Size, mut f: F) -> Self
    where
        F: FnMut(Position) -> T,
    {
        let mut data = Vec::with_capacity(size.height * size.width);
        for row in 0..size.height {
            for col in 0..size.width {
                data.push(f(Position { row, col }));
            }
        }
        Self {
            shape: Shape::from(size),
            data,
        }
    }

    /// Create owned surface from vector and sizes.
    pub fn from_vec(size: Size, data: Vec<T>) -> Self {
        assert!(size.height * size.width < data.len());
        Self {
            shape: Shape::from(size),
            data,
        }
    }

    pub fn to_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T> Surface for SurfaceOwned<T> {
    type Item = T;

    fn shape(&self) -> Shape {
        self.shape
    }

    fn data(&self) -> &[Self::Item] {
        &self.data
    }
}

impl<T> SurfaceMut for SurfaceOwned<T> {
    fn data_mut(&mut self) -> &mut [Self::Item] {
        &mut self.data
    }
}

impl<S> Surface for SurfaceOwnedView<S>
where
    S: Surface,
{
    type Item = S::Item;

    fn shape(&self) -> Shape {
        self.shape
    }

    fn data(&self) -> &[Self::Item] {
        self.inner.data()
    }
}

impl<S> SurfaceMut for SurfaceOwnedView<S>
where
    S: SurfaceMut,
{
    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.inner.data_mut()
    }
}

impl<'a, T: 'a> Surface for SurfaceView<'a, T> {
    type Item = T;

    fn shape(&self) -> Shape {
        self.shape
    }

    fn data(&self) -> &[Self::Item] {
        self.data
    }
}

impl<'a, T: 'a> Surface for SurfaceMutView<'a, T> {
    type Item = T;

    fn shape(&self) -> Shape {
        self.shape
    }

    fn data(&self) -> &[Self::Item] {
        self.data
    }
}

impl<'a, T: 'a> SurfaceMut for SurfaceMutView<'a, T> {
    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.data
    }
}

/// Everything that can be interpreted as a view bounds.
pub trait ViewBounds {
    /// Resolve bounds the same way python numpy does for ranges when indexing ndarrays
    ///
    /// Returns tuple with indices of the first element and the last + 1 element.
    fn view_bounds(self, size: usize) -> Option<(usize, usize)>;
}

impl ViewBounds for RangeFull {
    fn view_bounds(self, size: usize) -> Option<(usize, usize)> {
        range_bounds(self, size)
    }
}

macro_rules! impl_signed_ints(
    ($($int_type:ident),+) => {
        $(
            impl ViewBounds for $int_type {
                fn view_bounds(self, size: usize) -> Option<(usize, usize)> {
                    let size = size as $int_type;
                    if self < -size || self >= size {
                        None
                    } else {
                        let start = clamp(self + size, 0, 2 * size - 1) % size;
                        Some((start as usize, (start + 1) as usize))
                    }
                }
            }
        )+
    }
);
impl_signed_ints!(i8, i16, i32, i64, isize);

macro_rules! impl_unsigned_ints(
    ($($int_type:ident),+) => {
        $(
            impl ViewBounds for $int_type {
                fn view_bounds(self, size: usize) -> Option<(usize, usize)> {
                    let index = self as usize;
                    if index >= size {
                        None
                    } else {
                        Some((index, index + 1))
                    }
                }
            }
        )+
    }
);
impl_unsigned_ints!(u8, u16, u32, u64, usize);

macro_rules! impl_range_ints(
    ($($int_type:ident),+) => {
        $(
            impl ViewBounds for Range<$int_type> {
                fn view_bounds(self, size: usize) -> Option<(usize, usize)> {
                    range_bounds(
                        Range {
                            start: self.start as i64,
                            end: self.end as i64,
                        },
                        size,
                    )
                }
            }

            impl ViewBounds for RangeFrom<$int_type> {
                fn view_bounds(self, size: usize) -> Option<(usize, usize)> {
                    range_bounds(RangeFrom { start: self.start as i64 }, size)
                }
            }

            impl ViewBounds for RangeTo<$int_type> {
                fn view_bounds(self, size: usize) -> Option<(usize, usize)> {
                    range_bounds(RangeTo { end: self.end as i64 }, size)
                }
            }

            impl ViewBounds for RangeInclusive<$int_type> {
                fn view_bounds(self, size: usize) -> Option<(usize, usize)> {
                    let start = *self.start() as i64;
                    let end = *self.end() as i64;
                    range_bounds(start..=end, size)
                }
            }

            impl ViewBounds for RangeToInclusive<$int_type> {
                fn view_bounds(self, size: usize) -> Option<(usize, usize)> {
                    let end = self.end as i64;
                    range_bounds(..=end, size)
                }
            }
        )+
    }
);
impl_range_ints!(u8, i8, u16, i16, u32, i32, u64, i64, usize, isize);

fn range_bounds(bound: impl RangeBounds<i64>, size: usize) -> Option<(usize, usize)> {
    //  (index + size) % size - almost works
    //  0  1  2  3  4  5  6  7  8  9  0  1  2  3  4  5  6  7  8  9
    //-10 -9 -8 -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7  8  9
    let size = size as i64;
    if size == 0 {
        return None;
    }

    let (start, offset) = match bound.start_bound() {
        Bound::Unbounded => (0, 0),
        Bound::Included(start) => (*start, 0),
        Bound::Excluded(start) => (*start, 1),
    };
    let offset = if start >= size { 1 } else { offset };
    let start = clamp(start + size, 0, 2 * size - 1) % size + offset;

    let (end, offset) = match bound.end_bound() {
        Bound::Unbounded => (-1, 1),
        Bound::Included(end) => (*end, 1),
        Bound::Excluded(end) => (*end, 0),
    };
    let offset = if end >= size { 1 } else { offset };
    let end = clamp(end + size, 0, 2 * size - 1) % size + offset;

    if end <= start {
        None
    } else {
        Some((start as usize, end as usize))
    }
}

/// Construct new offset and shape for
pub fn view_shape<RS, CS>(shape: Shape, rows: RS, cols: CS) -> Shape
where
    RS: ViewBounds,
    CS: ViewBounds,
{
    match (
        cols.view_bounds(shape.width),
        rows.view_bounds(shape.height),
    ) {
        (Some((col_start, col_end)), Some((row_start, row_end))) => {
            let width = col_end - col_start;
            let height = row_end - row_start;
            let start = shape.offset(Position::new(row_start, col_start));
            let end = shape.offset(Position::new(row_end - 1, col_end));
            Shape {
                width,
                height,
                start,
                end,
                ..shape
            }
        }
        _ => Shape {
            height: 0,
            width: 0,
            row_stride: 0,
            col_stride: 0,
            start: 0,
            end: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_view_bounds() {
        assert_eq!((..).view_bounds(10), Some((0, 10)));
        assert_eq!((..-1).view_bounds(10), Some((0, 9)));
        assert_eq!((..=-1).view_bounds(10), Some((0, 10)));
        assert_eq!((-5..8).view_bounds(10), Some((5, 8)));
        assert_eq!((-10..).view_bounds(10), Some((0, 10)));
        assert_eq!((..20).view_bounds(10), Some((0, 10)));
        assert_eq!((10..20).view_bounds(10), None);
        assert_eq!((9..20).view_bounds(10), Some((9, 10)));
        assert_eq!((10..).view_bounds(10), None);
        assert_eq!(1.view_bounds(10), Some((1, 2)));
        assert_eq!((-1).view_bounds(10), Some((9, 10)));
        assert_eq!((-10).view_bounds(10), Some((0, 1)));
        assert_eq!((-11).view_bounds(10), None);
        assert_eq!(10.view_bounds(10), None);
        assert_eq!(10.view_bounds(0), None);
    }

    #[test]
    fn test_surface() {
        let mut surf = SurfaceOwned::new(Size::new(3, 4));
        assert!(!surf.is_empty());
        assert_eq!(surf.get(Position::new(2, 3)), Some(&0u8));
        assert_eq!(surf.get(Position::new(2, 4)), None);

        surf.view_mut(.., ..1).fill(1);
        surf.view_mut(1..2, ..).fill(2);
        surf.view_mut(.., -1..).fill(3);
        let reference: Vec<u8> = vec![
            1, 0, 0, 3, // 0
            2, 2, 2, 3, // 1
            1, 0, 0, 3, // 2
        ];
        assert_eq!(surf.iter().copied().collect::<Vec<_>>(), reference);

        let view = surf.view(..2, ..2);
        assert!(view.get(Position::new(0, 3)).is_none());
        assert!(view.get(Position::new(2, 0)).is_none());

        let mut view = surf.view_mut(..2, ..2);
        assert!(view.get_mut(Position::new(0, 3)).is_none());
        assert!(view.get_mut(Position::new(2, 0)).is_none());

        let view = surf.view(3..3, ..);
        assert!(view.is_empty());
        assert_eq!(view.get(Position::new(0, 0)), None);

        let view = surf.view(1..2, 1..3);
        assert_eq!(view.width(), 2);
        assert_eq!(view.height(), 1);
        assert_eq!(view.iter().count(), 2);
        assert_eq!(view.get(Position::new(0, 1)), Some(&2u8));
        assert_eq!(view.get(Position::new(0, 2)), None);
    }

    #[test]
    fn test_iter() {
        let mut data: Vec<usize> = Vec::new();
        data.resize_with(18, || 0);
        let mut surf = SurfaceOwned {
            shape: Shape {
                row_stride: 6,
                col_stride: 2,
                width: 3,
                height: 3,
                start: 0,
                end: data.len(),
            },
            data,
        };

        let mut view = surf.view_mut(.., 1..);
        assert_eq!(view.iter().count(), 6);
        assert_eq!(view.iter_mut().count(), 6);

        for (index, value) in view.iter_mut().enumerate() {
            *value = index + 1;
        }
        let reference = vec![
            0, 0, 1, 0, 2, 0, // 0
            0, 0, 3, 0, 4, 0, // 1
            0, 0, 5, 0, 6, 0, // 2
        ];
        assert_eq!(surf.data, reference);

        let mut surf = SurfaceOwned::new(Size::new(3, 7));
        surf.fill_with(|pos, _| (pos.row * 7 + pos.col) % 10);
        // 0 1 | 2 3 4 5 | 6
        // 7 8 | 9 0 1 2 | 3
        // 4 5 | 6 7 8 9 | 0
        let mut view = surf.view_mut(.., 2..-1);
        let mut iter = view.iter_mut();
        assert_eq!(iter.position(), Position::new(0, 0));
        assert_eq!(iter.next().cloned(), Some(2));
        assert_eq!(iter.position(), Position::new(0, 1));
        assert_eq!(iter.nth(1).cloned(), Some(4));
        assert_eq!(iter.position(), Position::new(0, 3));
        assert_eq!(iter.nth(6).cloned(), Some(7));
        assert_eq!(iter.position(), Position::new(2, 2));
        assert_eq!(iter.nth(1).cloned(), Some(9));
        assert_eq!(iter.position(), Position::new(3, 0));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.position(), Position::new(3, 0));

        let view = surf.view(1..2, 2..4);
        let mut iter = view.iter();
        assert_eq!(iter.next().cloned(), Some(9));
        assert_eq!(iter.position(), Position::new(0, 1));
        assert_eq!(iter.next().cloned(), Some(0));
        assert_eq!(iter.position(), Position::new(1, 0));
        assert_eq!(iter.next().cloned(), None);
        assert_eq!(iter.position(), Position::new(1, 0));
    }

    #[test]
    fn test_clear() {
        let mut surf = SurfaceOwned::new(Size::new(2, 2));
        surf.fill_with(|pos, _| pos.row * 2 + pos.col + 1);
        assert_eq!(surf.data(), &[1, 2, 3, 4]);
        surf.view_mut(.., 1..).clear();
        assert_eq!(surf.data(), &[1, 0, 3, 0]);
    }

    #[test]
    fn test_chains() {
        let mut surf: SurfaceOwned<()> = SurfaceOwned::new(Size::new(10, 10));
        let mut view = (&mut surf).view_owned(.., ..).view_owned(1..-1, ..);
        let mut view = view.view_mut(.., 1..-1);
        view.get_mut(Position::new(1, 1));
        assert_eq!(
            view.shape(),
            Shape {
                col_stride: 1,
                row_stride: 10,
                width: 8,
                height: 8,
                start: 11,
                end: 89,
            }
        );

        assert_eq!(
            (&surf).view(3.., 1..).view(..2, ..3).shape(),
            (&surf).view(3..5, 1..4).shape(),
        );
    }

    #[test]
    fn test_empty_surface() {
        let mut surf: SurfaceOwned<()> = SurfaceOwned::new(Size::new(0, 0));
        assert!(surf.iter().next().is_none());
        assert!(surf.iter_mut().next().is_none());

        let mut surf: SurfaceOwned<()> = SurfaceOwned::new(Size::new(0, 5));
        assert!(surf.iter().next().is_none());
        assert!(surf.iter_mut().next().is_none());

        let mut surf: SurfaceOwned<()> = SurfaceOwned::new(Size::new(5, 0));
        assert!(surf.iter().next().is_none());
        assert!(surf.iter_mut().next().is_none());
    }

    #[test]
    fn test_references() {
        let mut surf: SurfaceOwned<()> = SurfaceOwned::new(Size::new(3, 3));

        fn view<S: Surface>(s: S) {
            let _ = s.view(.., ..);
        }
        view(surf.clone());
        view(&surf);

        fn view_mut<S: SurfaceMut>(mut s: S) {
            let _ = s.view(.., ..);
            let _ = s.view_mut(.., ..);
        }
        view_mut(surf.clone());
        view(&mut surf);

        fn view_dyn(s: &dyn Surface<Item = ()>) {
            let _ = (&s).view(.., ..);
        }
        view_dyn(&surf);

        fn view_dyn_mut(mut s: &mut dyn SurfaceMut<Item = ()>) {
            let _ = (&s).view(.., ..);
            let _ = (&mut s).view_mut(.., ..);
        }
        view_dyn_mut(&mut surf);
    }
}
