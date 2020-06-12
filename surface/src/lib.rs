#![allow(clippy::iter_nth_zero)]
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::{
        Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
    },
    sync::Arc,
};

#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct Shape {
    /// Offsest of the first element.
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
    pub fn offset(&self, row: usize, col: usize) -> usize {
        self.start + row * self.row_stride + col * self.col_stride
    }

    /// Get row and column corresonding to nth element in row-major order
    #[inline]
    pub fn nth(&self, n: usize) -> Option<(usize, usize)> {
        if self.width == 0 || self.height == 0 {
            return None;
        }
        let row = n / self.width;
        let col = n - row * self.width;
        if row >= self.height || col >= self.width {
            None
        } else {
            Some((row, col))
        }
    }
}

pub trait Surface {
    type Item;

    /// Shape describes data layout inside `Self::data()` slice.
    fn shape(&self) -> Shape;

    /// Slice containing all the items
    ///
    /// **Note:** This slice contains all elements backed by parent object
    /// and elements should be accesed using the offeset calculcated by `Shape::offset`
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

    fn hash(&self) -> u64
    where
        Self::Item: Hash,
    {
        let mut hasher = DefaultHasher::new();
        hasher.write_usize(self.height());
        hasher.write_usize(self.width());
        for item in self.iter() {
            item.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Get immutable reference to the elemetn specified by row and column
    fn get(&self, row: usize, col: usize) -> Option<&Self::Item> {
        let shape = self.shape();
        if row >= shape.height || col >= shape.width {
            None
        } else {
            let data = self.data();
            data.get(shape.offset(row, col))
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
        F: FnMut(usize, usize, &Self::Item) -> T,
        Self: Sized,
    {
        let shape = self.shape();
        let data = self.data();
        SurfaceOwned::new_with(shape.height, shape.width, |row, col| {
            f(row, col, &data[shape.offset(row, col)])
        })
    }
}

pub trait SurfaceMut: Surface {
    /// Mutable slice containing all the items
    fn data_mut(&mut self) -> &mut [Self::Item];

    /// Get mutable reference to the elemetn specified by row and column
    fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut Self::Item> {
        let shape = self.shape();
        if row >= shape.height || col >= shape.width {
            None
        } else {
            let data = self.data_mut();
            data.get_mut(shape.offset(row, col))
        }
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

    /// Fill all elements of the surface with the copy of provided item.
    fn fill(&mut self, item: Self::Item)
    where
        Self::Item: Clone,
    {
        let shape = self.shape();
        let data = self.data_mut();
        for row in 0..shape.height {
            for col in 0..shape.width {
                data[shape.offset(row, col)] = item.clone();
            }
        }
    }

    /// Fill all the elments of the surface by colling af function.
    ///
    /// Function is called it row, column and the current item value as its arguments.
    fn fill_with<F>(&mut self, mut fill: F)
    where
        F: FnMut(usize, usize, Self::Item) -> Self::Item,
        Self::Item: Default,
        Self: Sized,
    {
        let shape = self.shape();
        let data = self.data_mut();
        let mut tmp = Self::Item::default();
        for row in 0..shape.height {
            for col in 0..shape.width {
                let offset = shape.offset(row, col);
                let item = std::mem::replace(&mut data[offset], tmp);
                tmp = std::mem::replace(&mut data[offset], fill(row, col, item));
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
                data[shape.offset(row, col)] = Default::default();
            }
        }
    }

    /// Insert items starting with the specified row and column.
    fn insert<IS>(&mut self, row: usize, col: usize, items: IS)
    where
        IS: IntoIterator<Item = Self::Item>,
        Self: Sized,
    {
        let index = row * self.width() + col;
        let mut iter = self.iter_mut();
        if index > 0 {
            iter.nth(index - 1);
        }
        for (src, dst) in items.into_iter().zip(iter) {
            *dst = src
        }
    }
}

pub struct SurfaceIter<'a, T> {
    index: usize,
    shape: Shape,
    data: &'a [T],
}

impl<'a, T: 'a> Iterator for SurfaceIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.nth(0)
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.index += n + 1;
        let (row, col) = self.shape.nth(self.index - 1)?;
        self.data.get(self.shape.offset(row, col))
    }
}

pub struct SurfaceMutIter<'a, T> {
    index: usize,
    shape: Shape,
    data: &'a mut [T],
}

impl<'a, T: 'a> Iterator for SurfaceMutIter<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        self.nth(0)
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.index += n + 1;
        let (row, col) = self.shape.nth(self.index - 1)?;
        let offset = self.shape.offset(row, col);

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

impl<'a, S> Surface for &'a S
where
    S: Surface + ?Sized,
{
    type Item = S::Item;

    fn shape(&self) -> Shape {
        (*self).shape()
    }

    fn data(&self) -> &[Self::Item] {
        (*self).data()
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

impl<T> Surface for Arc<dyn Surface<Item = T>> {
    type Item = T;

    fn shape(&self) -> Shape {
        (**self).shape()
    }

    fn data(&self) -> &[Self::Item] {
        (**self).data()
    }
}

#[derive(Clone)]
pub struct SurfaceOwned<T> {
    shape: Shape,
    data: Vec<T>,
}

#[derive(Clone)]
pub struct SurfaceOwnedView<S> {
    shape: Shape,
    inner: S,
}

#[derive(Clone)]
pub struct SurfaceView<'a, T> {
    shape: Shape,
    data: &'a [T],
}

pub struct SurfaceMutView<'a, T> {
    shape: Shape,
    data: &'a mut [T],
}

impl<T> SurfaceOwned<T> {
    pub fn new(height: usize, width: usize) -> Self
    where
        T: Default,
    {
        Self::new_with(height, width, |_, _| Default::default())
    }

    pub fn new_with<F>(height: usize, width: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let mut data = Vec::with_capacity(height * width);
        for row in 0..height {
            for col in 0..width {
                data.push(f(row, col));
            }
        }
        let shape = Shape {
            row_stride: width,
            col_stride: 1,
            height,
            width,
            start: 0,
            end: data.len(),
        };
        Self { shape, data }
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

/// Everything that can be intepreted as a view bounds.
pub trait ViewBounds {
    /// Resolve bounds the same way python numpy does for ranges when indexing ndarrays
    ///
    /// Returns tuple with indices of the first element and the last + 1 element.
    fn view_bounds(self, size: usize) -> Option<(usize, usize)>;
}

impl ViewBounds for Range<i32> {
    fn view_bounds(self, size: usize) -> Option<(usize, usize)> {
        range_bounds(self, size)
    }
}

impl ViewBounds for RangeFrom<i32> {
    fn view_bounds(self, size: usize) -> Option<(usize, usize)> {
        range_bounds(self, size)
    }
}

impl ViewBounds for RangeTo<i32> {
    fn view_bounds(self, size: usize) -> Option<(usize, usize)> {
        range_bounds(self, size)
    }
}

impl ViewBounds for RangeInclusive<i32> {
    fn view_bounds(self, size: usize) -> Option<(usize, usize)> {
        range_bounds(self, size)
    }
}

impl ViewBounds for RangeToInclusive<i32> {
    fn view_bounds(self, size: usize) -> Option<(usize, usize)> {
        range_bounds(self, size)
    }
}

impl ViewBounds for RangeFull {
    fn view_bounds(self, size: usize) -> Option<(usize, usize)> {
        range_bounds(self, size)
    }
}

impl ViewBounds for i32 {
    fn view_bounds(self, size: usize) -> Option<(usize, usize)> {
        let size = size as i32;
        if self < -size || self >= size {
            None
        } else {
            let start = clamp(self + size, 0, 2 * size - 1) % size;
            Some((start as usize, (start + 1) as usize))
        }
    }
}

fn range_bounds(bound: impl RangeBounds<i32>, size: usize) -> Option<(usize, usize)> {
    //  (index + size) % size - almost works
    //  0  1  2  3  4  5  6  7  8  9  0  1  2  3  4  5  6  7  8  9
    //-10 -9 -8 -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7  8  9
    let size = size as i32;

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

/// Construt new offset and shape for
fn view_shape<RS, CS>(shape: Shape, rows: RS, cols: CS) -> Shape
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
            let start = shape.offset(row_start, col_start);
            let end = shape.offset(row_end - 1, col_end);
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

fn clamp<T>(val: T, min: T, max: T) -> T
where
    T: PartialOrd,
{
    if val < min {
        min
    } else if val > max {
        max
    } else {
        val
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
    }

    #[test]
    fn test_surface() {
        let mut surf = SurfaceOwned::new(3, 4);
        assert!(!surf.is_empty());
        assert_eq!(surf.get(2, 3), Some(&0u8));
        assert_eq!(surf.get(2, 4), None);

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
        assert!(view.get(0, 3).is_none());
        assert!(view.get(2, 0).is_none());

        let mut view = surf.view_mut(..2, ..2);
        assert!(view.get_mut(0, 3).is_none());
        assert!(view.get_mut(2, 0).is_none());

        let view = surf.view(3..3, ..);
        assert!(view.is_empty());
        assert_eq!(view.get(0, 0), None);

        let view = surf.view(1..2, 1..3);
        assert_eq!(view.width(), 2);
        assert_eq!(view.height(), 1);
        assert_eq!(view.iter().count(), 2);
        assert_eq!(view.get(0, 1), Some(&2u8));
        assert_eq!(view.get(0, 2), None);
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

        let mut surf = SurfaceOwned::new(3, 7);
        surf.fill_with(|row, col, _| (row * 7 + col) % 10);
        // 0 1 | 2 3 4 5 | 6
        // 7 8 | 9 0 1 2 | 3
        // 4 5 | 6 7 8 9 | 1
        let mut view = surf.view_mut(.., 2..-1);
        let mut iter = view.iter_mut();
        assert_eq!(iter.next().cloned(), Some(2));
        assert_eq!(iter.nth(1).cloned(), Some(4));
        assert_eq!(iter.nth(6).cloned(), Some(7));
        assert_eq!(iter.nth(1).cloned(), Some(9));
        assert_eq!(iter.next(), None);

        let view = surf.view(1..2, 2..4);
        let mut iter = view.iter();
        assert_eq!(iter.next().cloned(), Some(9));
        assert_eq!(iter.next().cloned(), Some(0));
        assert_eq!(iter.next().cloned(), None);
    }

    #[test]
    fn test_clear() {
        let mut surf = SurfaceOwned::new(2, 2);
        surf.fill_with(|r, c, _| r * 2 + c + 1);
        assert_eq!(surf.data(), &[1, 2, 3, 4]);
        surf.view_mut(.., 1..).clear();
        assert_eq!(surf.data(), &[1, 0, 3, 0]);
    }

    #[test]
    fn test_chains() {
        let mut surf: SurfaceOwned<()> = SurfaceOwned::new(10, 10);
        let mut view = (&mut surf).view_owned(.., ..).view_owned(1..-1, ..);
        let mut view = view.view_mut(.., 1..-1);
        view.get_mut(1, 1);
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
        let mut surf: SurfaceOwned<()> = SurfaceOwned::new(0, 0);
        assert!(surf.iter().next().is_none());
        assert!(surf.iter_mut().next().is_none());
    }

    #[test]
    fn test_references() {
        let mut surf: SurfaceOwned<()> = SurfaceOwned::new(3, 3);

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