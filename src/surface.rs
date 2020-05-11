use std::ops::{Bound, Range, RangeBounds};

#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct Shape {
    pub row_stride: usize,
    pub col_stride: usize,
    pub height: usize,
    pub width: usize,
}

impl Shape {
    #[inline]
    pub fn index(&self, row: usize, col: usize) -> usize {
        row * self.row_stride + col * self.col_stride
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

    /// Get item at specified `row` and `col`
    fn get(&self, row: usize, col: usize) -> Option<&Self::Item> {
        self.data().get(self.shape().index(row, col))
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
            let index = self.shape.index(self.row, self.col);
            self.col += 1;
            if self.col >= self.shape.width {
                self.col = 0;
                self.row += 1;
            }
            Some(&self.data[index])
        }
    }
}

pub trait ViewMut: View {
    /// Mutable slice containing all the items
    ///
    /// **Note:** This slice also include elements not belonging to this view,
    /// use `Self::get_mut` function instead to access items.
    fn data_mut(&mut self) -> &mut [Self::Item];

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

    /// Fill view with the `fill` function.
    fn fill<F>(&mut self, mut fill: F)
    where
        F: FnMut(usize, usize, Self::Item) -> Self::Item,
        Self::Item: Default,
    {
        let shape = self.shape();
        let data = self.data_mut();
        let mut tmp = Self::Item::default();
        for row in 0..shape.height {
            for col in 0..shape.width {
                let index = shape.index(row, col);
                let item = std::mem::replace(&mut data[index], tmp);
                tmp = std::mem::replace(&mut data[index], fill(row, col, item));
            }
        }
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
                end: start + shape.index(height - 1, width),
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

        surf.view_mut(.., ..1).fill(|_, _, _| 1);
        surf.view_mut(1..2, ..).fill(|_, _, _| 2);
        surf.view_mut(.., -1..).fill(|_, _, _| 3);
        let reference: Vec<u8> = vec![1, 0, 0, 3, 2, 2, 2, 3, 1, 0, 0, 3];
        assert_eq!(surf.iter().copied().collect::<Vec<_>>(), reference);

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
}
