use std::ops::{Bound, Range, RangeBounds};

#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct Shape {
    pub row_stride: usize,
    pub col_stride: usize,
    pub height: usize,
    pub width: usize,
}

impl Shape {
    /// Convert row and col to offset.
    #[inline]
    pub fn offset(&self, row: usize, col: usize) -> usize {
        row * self.row_stride + col * self.col_stride
    }

    /// Givern row and column move `nth` steps in row major order
    pub fn nth(&self, row: usize, col: usize, nth: usize) -> (usize, usize) {
        let index = row * self.width + col + nth;
        let row = index / self.width;
        let col = index - row * self.width;
        (row, col)
    }
}

pub trait View {
    type Item;

    /// Shape describes data layout inside `Self::data()` slice.
    fn shape(&self) -> Shape;

    /// Slice containing all the items
    ///
    /// **Note:** This slice also include elements not belonging to this view,
    /// use `Self::get` function instead to access items.
    fn data(&self) -> &[Self::Item];
}

pub trait ViewExt: View {
    /// Returns `true` if the view contains no elements
    fn is_empty(&self) -> bool {
        self.data().is_empty()
    }

    /// Height of the view
    fn height(&self) -> usize {
        self.shape().height
    }

    /// Width of the view
    fn width(&self) -> usize {
        self.shape().width
    }

    /// Get a reference to item at specified `row` and `col`
    fn get(&self, row: usize, col: usize) -> Option<&Self::Item> {
        let shape = self.shape();
        if row < shape.height && col < shape.width {
            self.data().get(shape.offset(row, col))
        } else {
            None
        }
    }

    /// Create read only sub-view restricted by `rows` and `cols` bounds.
    fn view<RS, CS>(&self, rows: RS, cols: CS) -> SurfaceView<'_, Self::Item>
    where
        RS: RangeBounds<i32>,
        CS: RangeBounds<i32>,
    {
        let (range, shape) = view_shape(self.shape(), rows, cols);
        let data = &self.data()[range];
        SurfaceView { shape, data }
    }

    /// Iterator over all items
    fn iter(&self) -> ViewIter<'_, Self::Item> {
        ViewIter {
            row: 0,
            col: 0,
            shape: self.shape(),
            data: self.data(),
        }
    }
}

impl<V: View + ?Sized> ViewExt for V {}

pub struct ViewIter<'a, Item> {
    row: usize,
    col: usize,
    shape: Shape,
    data: &'a [Item],
}

impl<'a, Item> Iterator for ViewIter<'a, Item> {
    type Item = &'a Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row >= self.shape.height || self.col >= self.shape.width {
            None
        } else {
            let offset = self.shape.offset(self.row, self.col);
            self.col += 1;
            if self.col >= self.shape.width {
                self.col = 0;
                self.row += 1;
            }
            self.data.get(offset)
        }
    }
}

impl<'a, T> View for &'a T
where
    T: View + ?Sized,
{
    type Item = T::Item;

    fn shape(&self) -> Shape {
        (**self).shape()
    }

    fn data(&self) -> &[Self::Item] {
        (**self).data()
    }
}

impl<'a, T> View for &'a mut T
where
    T: View + ?Sized,
{
    type Item = T::Item;

    fn shape(&self) -> Shape {
        (**self).shape()
    }

    fn data(&self) -> &[Self::Item] {
        (**self).data()
    }
}

pub trait ViewMut: View {
    /// Mutable slice containing all the items
    ///
    /// **Note:** This slice also include elements not belonging to this view,
    /// use `Self::get_mut` function instead to access items.
    fn data_mut(&mut self) -> &mut [Self::Item];

    /// Get a mutable reference to item at specified `row` and `col`
    fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut Self::Item> {
        let shape = self.shape();
        if row < shape.height && col < shape.width {
            self.data_mut().get_mut(shape.offset(row, col))
        } else {
            None
        }
    }
}

pub trait ViewMutExt: ViewMut {
    /// Create mutable sub-view restricted by `rows` and `cols` bounds.
    fn view_mut<RS, CS>(&mut self, rows: RS, cols: CS) -> SurfaceViewMut<'_, Self::Item>
    where
        RS: RangeBounds<i32>,
        CS: RangeBounds<i32>,
    {
        let (range, shape) = view_shape(self.shape(), rows, cols);
        let data = &mut self.data_mut()[range];
        SurfaceViewMut { shape, data }
    }

    fn iter_mut(&mut self) -> ViewMutIter<'_, Self::Item> {
        ViewMutIter {
            row: 0,
            col: 0,
            shape: self.shape(),
            data: self.data_mut(),
        }
    }

    /// Fill view with the `fill` function.
    fn fill_with<F>(&mut self, mut fill: F)
    where
        F: FnMut(usize, usize, Self::Item) -> Self::Item,
        Self::Item: Default,
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

    /// Fill view with provided item.
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

    /// Insert items in in row major order starting at specified row and column.
    fn insert<IS>(&mut self, row: usize, col: usize, items: IS)
    where
        IS: IntoIterator<Item = Self::Item>,
    {
        let index = row * self.shape().width + col;
        let mut iter = self.iter_mut();
        if index > 0 {
            iter.nth(index - 1);
        }
        for (src, dst) in items.into_iter().zip(iter) {
            *dst = src
        }
    }

    /// Clear view by replacing all items with default value.
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
}

impl<V: ViewMut + ?Sized> ViewMutExt for V {}

pub struct ViewMutIter<'a, Item: 'a> {
    row: usize,
    col: usize,
    shape: Shape,
    data: &'a mut [Item],
}

impl<'a, Item: 'a> Iterator for ViewMutIter<'a, Item> {
    type Item = &'a mut Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.nth(0)
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if self.data.is_empty() {
            return None;
        }

        let offset = self.shape.offset(self.row, self.col);
        let (r_cut, c_cut) = self.shape.nth(self.row, self.col, n + 1);
        let o_cut = self.shape.offset(r_cut, c_cut) - offset;
        let o_val = if n == 0 {
            0
        } else {
            let (r_val, c_val) = self.shape.nth(self.row, self.col, n);
            self.shape.offset(r_val, c_val) - offset
        };

        let data = std::mem::replace(&mut self.data, &mut []);
        let (head, data) = data.split_at_mut(std::cmp::min(data.len(), o_cut));
        let result = head.get_mut(o_val);

        self.row = r_cut;
        self.col = c_cut;
        if !result.is_none() {
            self.data = data;
        }
        result
    }
}

impl<'a, T> ViewMut for &'a mut T
where
    T: ViewMut + ?Sized,
{
    fn data_mut(&mut self) -> &mut [Self::Item] {
        (**self).data_mut()
    }
}

#[derive(Clone)]
pub struct Surface<Item> {
    shape: Shape,
    data: Vec<Item>,
}

impl<Item> Surface<Item> {
    pub fn new(height: usize, width: usize) -> Self
    where
        Item: Default,
    {
        let mut data = Vec::new();
        data.resize_with(height * width, Default::default);
        Self {
            shape: Shape {
                row_stride: width,
                col_stride: 1,
                height,
                width,
            },
            data,
        }
    }
}

impl<Item> View for Surface<Item> {
    type Item = Item;

    fn shape(&self) -> Shape {
        self.shape
    }

    fn data(&self) -> &[Self::Item] {
        &self.data
    }
}

impl<Item> ViewMut for Surface<Item> {
    fn data_mut(&mut self) -> &mut [Self::Item] {
        &mut self.data
    }
}

#[derive(Clone)]
pub struct SurfaceView<'a, Item> {
    shape: Shape,
    data: &'a [Item],
}

impl<'a, Item> View for SurfaceView<'a, Item> {
    type Item = Item;

    fn shape(&self) -> Shape {
        self.shape
    }

    fn data(&self) -> &[Self::Item] {
        &self.data
    }
}

pub struct SurfaceViewMut<'a, Item> {
    shape: Shape,
    data: &'a mut [Item],
}

impl<'a, Item> View for SurfaceViewMut<'a, Item> {
    type Item = Item;

    fn shape(&self) -> Shape {
        self.shape
    }

    fn data(&self) -> &[Self::Item] {
        &self.data
    }
}

impl<'a, Item> ViewMut for SurfaceViewMut<'a, Item> {
    fn data_mut(&mut self) -> &mut [Self::Item] {
        &mut self.data
    }
}

#[inline]
fn clip<T>(val: T, min: T, max: T) -> T
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
    let start = clip(start + size, 0, 2 * size - 1) % size + offset;

    let (end, offset) = match bound.end_bound() {
        Bound::Unbounded => (-1, 1),
        Bound::Included(end) => (*end, 1),
        Bound::Excluded(end) => (*end, 0),
    };
    let offset = if end >= size { 1 } else { offset };
    let end = clip(end + size, 0, 2 * size - 1) % size + offset;

    if end <= start {
        None
    } else {
        Some((start as usize, end as usize))
    }
}

/// Construt new offset and shape for
fn view_shape<RS, CS>(shape: Shape, rows: RS, cols: CS) -> (Range<usize>, Shape)
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
            let shape = Shape {
                width,
                height,
                ..shape
            };
            let start = col_start * shape.col_stride + row_start * shape.row_stride;
            let range = Range {
                start,
                end: start + shape.offset(height - 1, width),
            };
            (range, shape)
        }
        _ => {
            let shape = Shape {
                height: 0,
                width: 0,
                row_stride: 0,
                col_stride: 0,
            };
            let range = Range { start: 0, end: 0 };
            (range, shape)
        }
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
        let mut surf: Surface<u8> = Surface::new(3, 4);
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
    fn test_view_mut_iter() {
        let mut data: Vec<usize> = Vec::new();
        data.resize_with(18, || 0);
        let mut surf = SurfaceViewMut {
            data: &mut data,
            shape: Shape {
                row_stride: 6,
                col_stride: 2,
                width: 3,
                height: 3,
            },
        };

        let mut view = surf.view_mut(.., 1..);
        assert_eq!(view.iter().count(), 6);
        // assert_eq!(view.iter_mut().count(), 6);

        for (index, value) in view.iter_mut().enumerate() {
            *value = index + 1;
        }
        let reference = vec![
            0, 0, 1, 0, 2, 0, // 0
            0, 0, 3, 0, 4, 0, // 1
            0, 0, 5, 0, 6, 0, // 2
        ];
        assert_eq!(data, reference);

        let mut surf: Surface<usize> = Surface::new(3, 7);
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
    }

    #[test]
    fn test_view_dyn() {
        let mut surf: Surface<usize> = Surface::new(1, 1);

        fn is_view_dyn(view: &dyn View<Item = usize>) {
            let _ = view.view(.., ..);
        }
        is_view_dyn(&surf);
        is_view_dyn(&surf.view(.., ..));

        fn is_view_mut_dyn(view: &mut dyn ViewMut<Item = usize>) {
            let _ = view.view_mut(.., ..);
        }
        is_view_mut_dyn(&mut surf);
        is_view_mut_dyn(&mut surf.view_mut(.., ..));

        fn is_view<V: View>(view: V) {
            let _ = view.view(.., ..);
        }
        is_view(surf.clone());
        is_view(&surf);

        fn is_view_mut<V: ViewMut>(mut view: V) {
            let _ = view.view_mut(.., ..);
        }
        is_view_mut(surf.clone());
        is_view_mut(&mut surf);
    }

    #[test]
    fn terst_clear() {
        let mut surf: Surface<usize> = Surface::new(2, 2);
        surf.fill_with(|r, c, _| r * 2 + c + 1);
        assert_eq!(surf.data(), &[1, 2, 3, 4]);
        surf.view_mut(.., 1..).clear();
        assert_eq!(surf.data(), &[1, 0, 3, 0]);
    }
}
