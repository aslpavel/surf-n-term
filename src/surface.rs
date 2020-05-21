use std::ops::{Bound, RangeBounds};

#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct Shape {
    /// Offsest of the first element.
    pub start: usize,
    /// Offset of the last + 1 element.
    pub end: usize,
    /// Width of the storage
    pub width: usize,
    /// Height of the storage
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

pub trait Storage {
    type Item;

    /// Shape describes data layout inside `Self::data()` slice.
    fn shape(&self) -> Shape;

    /// Slice containing all the items
    ///
    /// **Note:** This slice contains all elements backed by parent object
    /// and elements should accesed using offeset calculcated by `Shape::offset`
    fn data(&self) -> &[Self::Item];
}

pub trait StorageMut: Storage {
    /// Mutable slice containing all the items
    fn data_mut(&mut self) -> &mut [Self::Item];
}

impl<'a, S> Storage for &'a S
where
    S: Storage + ?Sized,
{
    type Item = S::Item;

    fn shape(&self) -> Shape {
        (*self).shape()
    }

    fn data(&self) -> &[Self::Item] {
        (*self).data()
    }
}

impl<'a, S> Storage for &'a mut S
where
    S: Storage + ?Sized,
{
    type Item = S::Item;

    fn shape(&self) -> Shape {
        (**self).shape()
    }

    fn data(&self) -> &[Self::Item] {
        (**self).data()
    }
}

impl<'a, S> StorageMut for &'a mut S
where
    S: StorageMut + ?Sized,
{
    fn data_mut(&mut self) -> &mut [Self::Item] {
        (**self).data_mut()
    }
}

pub struct Owned<T> {
    shape: Shape,
    data: Vec<T>,
}

pub struct Shared<S> {
    shape: Shape,
    inner: S,
}

impl<T> Storage for Owned<T> {
    type Item = T;

    fn shape(&self) -> Shape {
        self.shape
    }

    fn data(&self) -> &[Self::Item] {
        &self.data
    }
}

impl<T> StorageMut for Owned<T> {
    fn data_mut(&mut self) -> &mut [Self::Item] {
        &mut self.data
    }
}

impl<S> Storage for Shared<S>
where
    S: Storage,
{
    type Item = S::Item;

    fn shape(&self) -> Shape {
        self.shape
    }

    fn data(&self) -> &[Self::Item] {
        self.inner.data()
    }
}

impl<S> StorageMut for Shared<S>
where
    S: StorageMut,
{
    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.inner.data_mut()
    }
}

pub struct Surface<S> {
    storage: S,
}

pub type SurfaceOwned<T> = Surface<Owned<T>>;

impl<T> Surface<Owned<T>>
where
    T: Default,
{
    pub fn new(height: usize, width: usize) -> Self {
        let mut data = Vec::new();
        data.resize_with(height * width, Default::default);
        let shape = Shape {
            row_stride: width,
            col_stride: 1,
            height,
            width,
            start: 0,
            end: data.len(),
        };
        Self {
            storage: Owned { shape, data },
        }
    }
}

impl<S> Surface<S>
where
    S: Storage,
{
    pub fn is_empty(&self) -> bool {
        let shape = self.storage.shape();
        shape.start >= shape.end
    }

    pub fn height(&self) -> usize {
        self.storage.shape().height
    }

    pub fn width(&self) -> usize {
        self.storage.shape().width
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&S::Item> {
        let shape = self.storage.shape();
        if row >= shape.height || col >= shape.width {
            None
        } else {
            let data = self.storage.data();
            data.get(shape.offset(row, col))
        }
    }

    pub fn by_ref(&self) -> Surface<&S> {
        Surface {
            storage: &self.storage,
        }
    }

    pub fn by_ref_dyn(&self) -> Surface<&dyn Storage<Item = S::Item>> {
        Surface {
            storage: &self.storage,
        }
    }

    /// Iterator over all elements of the surface in the row-major order.
    pub fn iter(&self) -> SurfaceIter<'_, S::Item> {
        SurfaceIter {
            index: 0,
            shape: self.storage.shape(),
            data: self.storage.data(),
        }
    }

    /// Create a sub-surface restricted by `rows` and `cols` bounds.
    ///
    /// This method consumes current surface. But it also possible to use
    /// surface multiple times if by using `Suface::by_ref*` methods.
    pub fn view<RS, CS>(self, rows: RS, cols: CS) -> Surface<Shared<S>>
    where
        RS: RangeBounds<i32>,
        CS: RangeBounds<i32>,
    {
        let storage = Shared {
            shape: view_shape(self.storage.shape(), rows, cols),
            inner: self.storage,
        };
        Surface { storage }
    }
}

impl<S> Surface<S>
where
    S: StorageMut,
{
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut S::Item> {
        let shape = self.storage.shape();
        if row >= shape.height || col >= shape.width {
            None
        } else {
            let data = self.storage.data_mut();
            data.get_mut(shape.offset(row, col))
        }
    }

    pub fn by_ref_mut(&mut self) -> Surface<&mut S> {
        Surface {
            storage: &mut self.storage,
        }
    }

    pub fn by_ref_mut_dyn(&mut self) -> Surface<&mut dyn StorageMut<Item = S::Item>> {
        Surface {
            storage: &mut self.storage,
        }
    }

    /// Mutable iterator over all elements of the surface in the row-major order.
    pub fn iter_mut(&mut self) -> SurfaceIterMut<'_, S::Item> {
        SurfaceIterMut {
            index: 0,
            shape: self.storage.shape(),
            data: self.storage.data_mut(),
        }
    }

    /// Fill all elements of the surface with the copy of provided item.
    pub fn fill(&mut self, item: S::Item)
    where
        S::Item: Clone,
    {
        let shape = self.storage.shape();
        let data = self.storage.data_mut();
        for row in 0..shape.height {
            for col in 0..shape.width {
                data[shape.offset(row, col)] = item.clone();
            }
        }
    }

    /// Fill all the elments of the surface by colling af function.
    ///
    /// Function is called it row, column and the current item value as its arguments.
    pub fn fill_with<F>(&mut self, mut fill: F)
    where
        F: FnMut(usize, usize, S::Item) -> S::Item,
        S::Item: Default,
    {
        let shape = self.storage.shape();
        let data = self.storage.data_mut();
        let mut tmp = S::Item::default();
        for row in 0..shape.height {
            for col in 0..shape.width {
                let offset = shape.offset(row, col);
                let item = std::mem::replace(&mut data[offset], tmp);
                tmp = std::mem::replace(&mut data[offset], fill(row, col, item));
            }
        }
    }

    /// Fill all the element of the surface with default value.
    pub fn clear(&mut self)
    where
        S::Item: Default,
    {
        let shape = self.storage.shape();
        let data = self.storage.data_mut();
        for row in 0..shape.height {
            for col in 0..shape.width {
                data[shape.offset(row, col)] = Default::default();
            }
        }
    }

    pub fn insert<IS>(&mut self, row: usize, col: usize, items: IS)
    where
        IS: IntoIterator<Item = S::Item>,
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

pub struct SurfaceIterMut<'a, T> {
    index: usize,
    shape: Shape,
    data: &'a mut [T],
}

impl<'a, T: 'a> Iterator for SurfaceIterMut<'a, T> {
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
            let item = unsafe { &mut *ptr.offset(offset as isize) };
            Some(item)
        }
    }
}

#[inline]
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

/// Resolve bounds the same way python or numpy does for ranges when indexing arrays
pub fn view_bound(bound: impl RangeBounds<i32>, size: usize) -> Option<(usize, usize)> {
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
    RS: RangeBounds<i32>,
    CS: RangeBounds<i32>,
{
    match (
        view_bound(cols, shape.width),
        view_bound(rows, shape.height),
    ) {
        (Some((col_start, col_end)), Some((row_start, row_end))) => {
            let width = col_end - col_start;
            let height = row_end - row_start;
            let start = col_start * shape.col_stride + row_start * shape.row_stride;
            let end = start + shape.offset(height - 1, width);
            let shape = Shape {
                width,
                height,
                start,
                end,
                ..shape
            };
            shape
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
    fn test_view_bound() {
        assert_eq!(view_bound(.., 10), Some((0, 10)));
        assert_eq!(view_bound(..-1, 10), Some((0, 9)));
        assert_eq!(view_bound(..=-1, 10), Some((0, 10)));
        assert_eq!(view_bound(-5..8, 10), Some((5, 8)));
        assert_eq!(view_bound(-10.., 10), Some((0, 10)));
        assert_eq!(view_bound(..20, 10), Some((0, 10)));
        assert_eq!(view_bound(10..20, 10), None);
        assert_eq!(view_bound(9..20, 10), Some((9, 10)));
    }

    #[test]
    fn test_surface() {
        let mut surf = Surface::new(3, 4);
        assert!(!surf.is_empty());
        assert_eq!(surf.get(2, 3), Some(&0u8));
        assert_eq!(surf.get(2, 4), None);

        surf.by_ref_mut().view(.., ..1).fill(1);
        surf.by_ref_mut().view(1..2, ..).fill(2);
        surf.by_ref_mut().view(.., -1..).fill(3);
        let reference: Vec<u8> = vec![
            1, 0, 0, 3, // 0
            2, 2, 2, 3, // 1
            1, 0, 0, 3, // 2
        ];
        assert_eq!(surf.iter().copied().collect::<Vec<_>>(), reference);

        let view = surf.by_ref().view(..2, ..2);
        assert!(view.get(0, 3).is_none());
        assert!(view.get(2, 0).is_none());

        let mut view = surf.by_ref_mut().view(..2, ..2);
        assert!(view.get_mut(0, 3).is_none());
        assert!(view.get_mut(2, 0).is_none());

        let view = surf.by_ref().view(3..3, ..);
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
    fn test_view_mut_iter() {
        let mut data: Vec<usize> = Vec::new();
        data.resize_with(18, || 0);
        let mut surf = Surface {
            storage: Owned {
                shape: Shape {
                    row_stride: 6,
                    col_stride: 2,
                    width: 3,
                    height: 3,
                    start: 0,
                    end: data.len(),
                },
                data,
            },
        };

        let mut view = surf.by_ref_mut().view(.., 1..);
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
        assert_eq!(surf.storage.data, reference);

        let mut surf = Surface::new(3, 7);
        surf.fill_with(|row, col, _| (row * 7 + col) % 10);
        // 0 1 | 2 3 4 5 | 6
        // 7 8 | 9 0 1 2 | 3
        // 4 5 | 6 7 8 9 | 1
        let mut view = surf.by_ref_mut().view(.., 2..-1);
        let mut iter = view.iter_mut();
        assert_eq!(iter.next().cloned(), Some(2));
        assert_eq!(iter.nth(1).cloned(), Some(4));
        assert_eq!(iter.nth(6).cloned(), Some(7));
        assert_eq!(iter.nth(1).cloned(), Some(9));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_clear() {
        let mut surf = Surface::new(2, 2);
        surf.fill_with(|r, c, _| r * 2 + c + 1);
        assert_eq!(surf.storage.data(), &[1, 2, 3, 4]);
        surf.by_ref_mut().view(.., 1..).clear();
        assert_eq!(surf.storage.data(), &[1, 0, 3, 0]);
    }

    #[test]
    fn test_chains() {
        let surf: Surface<Owned<()>> = Surface::new(10, 10);
        let mut view = surf.view(.., ..).view(1..-1, ..).view(.., 1..-1);
        view.get_mut(1, 1);
        assert_eq!(
            view.storage.shape(),
            Shape {
                col_stride: 1,
                row_stride: 10,
                width: 8,
                height: 8,
                start: 1,
                end: 89,
            }
        )
    }

    #[test]
    fn test_empty_surface() {
        let mut surf: Surface<Owned<()>> = Surface::new(0, 0);
        assert!(surf.iter().next().is_none());
        assert!(surf.iter_mut().next().is_none());
    }
}
