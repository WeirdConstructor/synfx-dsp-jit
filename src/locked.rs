// Copyright (c) 2022 Weird Constructor <weirdconstructor@gmail.com>
// This file is a part of synfx-dsp-jit. Released under GPL-3.0-or-later.
// See README.md and COPYING for details.

/// Represents a type that I can get a non mutable pointer to and a number of elements
/// that it points to:
pub trait Pointable<E> {
    /// Hands out a mutable pointer to the underlying memory:
    fn as_ptr(&self) -> *const E;
    /// The number of the elements pointed to by [MutPointable::as_mut_ptr]
    fn len(&self) -> usize;
}

impl<E> Pointable<E> for Vec<E> {
    fn len(&self) -> usize {
        self.len()
    }

    fn as_ptr(&self) -> *const E {
        self.as_ptr()
    }
}

impl<E> Pointable<E> for std::sync::Arc<Vec<E>> {
    fn len(&self) -> usize {
        self.as_ref().len()
    }

    fn as_ptr(&self) -> *const E {
        self.as_ref().as_ptr()
    }
}

/// Represents a type that I can get a mutable pointer to and a number of elements
/// that it points to:
pub trait MutPointable<E> {
    /// Hands out a mutable pointer to the underlying memory:
    fn as_mut_ptr(&mut self) -> *mut E;
    /// The number of the elements pointed to by [MutPointable::as_mut_ptr]
    fn len(&self) -> usize;
}

impl<E> MutPointable<E> for Vec<E> {
    fn len(&self) -> usize {
        self.len()
    }

    fn as_mut_ptr(&mut self) -> *mut E {
        self.as_mut_ptr()
    }
}

/// This type locks up a Vec<> after creation and gives you a pointer to
/// a vector of mutable pointers: `*const *mut T`.
///
/// It locks away the interior [std::vec::Vec] instances, so that they can't be
/// accidentally changed while you are messing around with the pointers.
///
/// See also [LockedPtrs] for a mutable alternative to this type.
///
/// The type `T` must implement the [MutPointable] trait.
///
///```
/// use synfx_dsp_jit::locked::*;
///
/// struct MyFFIStuff {
///     data: LockedMutPtrs<Vec<f64>, f64>,
/// }
///
/// let ffistuff = MyFFIStuff {
///     data: LockedMutPtrs::new(vec![vec![0.0; 10], vec![0.2; 30]]),
/// };
///
/// // We are guaranteed here, that the pointers here will not be
/// // invalidated until MyFFIStuff is dropped!
/// let pointers = ffistuff.data.pointers();
///
/// unsafe {
///     (*pointers[0]) = 10.0;
///     (*pointers[0].offset(1)) = 11.0;
///     (*pointers[1]) = 20.0;
/// };
///```
pub struct LockedMutPtrs<T, E>
where
    T: MutPointable<E>,
{
    data: Vec<T>,
    pointers: Vec<*mut E>,
    lens: Vec<u64>,
    phantom: std::marker::PhantomData<E>,
}

impl<T, E> LockedMutPtrs<T, E>
where
    T: MutPointable<E>,
{
    pub fn new(mut v: Vec<T>) -> Self {
        let mut pointers: Vec<*mut E> = vec![];
        let mut lens = vec![];
        for elem in v.iter_mut() {
            pointers.push(elem.as_mut_ptr());
            lens.push(elem.len() as u64);
        }

        Self { data: v, pointers, lens, phantom: std::marker::PhantomData }
    }

    /// Swaps out one element in the locked vector.
    ///
    /// # Safety
    ///
    /// You must not call this function while you still have pointers handed out.
    /// These pointers will be invaldiated by this function!
    pub unsafe fn swap_element(&mut self, idx: usize, elem: T) -> Result<T, T> {
        if idx >= self.pointers.len() {
            Err(elem)
        } else {
            if elem.len() == 0 {
                return Err(elem);
            }

            let ret = std::mem::replace(&mut self.data[idx], elem);
            self.pointers[idx] = self.data[idx].as_mut_ptr();
            self.lens[idx] = self.data[idx].len() as u64;
            Ok(ret)
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.pointers.len()
    }

    #[inline]
    pub fn element_len(&self, idx: usize) -> usize {
        if idx >= self.data.len() {
            return 0;
        }
        self.data[idx].len()
    }

    #[inline]
    pub fn lens(&self) -> &[u64] {
        self.lens.as_ref()
    }

    #[inline]
    pub fn pointers(&self) -> &[*mut E] {
        self.pointers.as_ref()
    }
}

/// This type locks up a Vec<> after creation and gives you a pointer to
/// a vector of pointers: `*const *const T`.
///
/// It locks away the interior [std::vec::Vec] instances, so that they can't be
/// accidentally changed while you are messing around with the pointers.
///
/// The type `T` must implement the [Pointable] trait.
///
/// See also [LockedMutPtrs] for a mutable alternative to this type.
///
///```
/// use synfx_dsp_jit::locked::*;
///
/// struct MyFFIStuff {
///     data: LockedPtrs<Vec<i64>, i64>,
/// }
///
/// let ffistuff = MyFFIStuff {
///     data: LockedPtrs::new(vec![vec![1; 10], vec![2; 30]]),
/// };
///
/// // We are guaranteed here, that the pointers here will not be
/// // invalidated until MyFFIStuff is dropped!
/// let pointers = ffistuff.data.pointers();
///
/// unsafe {
///     assert_eq!((*pointers[0]), 1);
///     assert_eq!((*pointers[1]), 2);
///     assert_eq!((*pointers[0].offset(5)), 1);
///     assert_eq!((*pointers[1].offset(5)), 2);
/// };
///```
pub struct LockedPtrs<T, E>
where
    T: Pointable<E>,
{
    data: Vec<T>,
    pointers: Vec<*const E>,
    lens: Vec<u64>,
    phantom: std::marker::PhantomData<E>,
}

impl<T, E> LockedPtrs<T, E>
where
    T: Pointable<E>,
{
    pub fn new(v: Vec<T>) -> Self {
        let mut pointers: Vec<*const E> = vec![];
        let mut lens = vec![];
        for elem in v.iter() {
            pointers.push(elem.as_ptr());
            lens.push(elem.len() as u64);
        }

        Self { data: v, pointers, lens, phantom: std::marker::PhantomData }
    }

    /// Swaps out one element in the locked vector.
    ///
    /// # Safety
    ///
    /// You must not call this function while you still have pointers handed out.
    /// These pointers will be invaldiated by this function!
    pub unsafe fn swap_element(&mut self, idx: usize, elem: T) -> Result<T, T> {
        if idx >= self.pointers.len() {
            Err(elem)
        } else {
            if elem.len() == 0 {
                return Err(elem);
            }

            let ret = std::mem::replace(&mut self.data[idx], elem);
            self.pointers[idx] = self.data[idx].as_ptr();
            self.lens[idx] = self.data[idx].len() as u64;
            Ok(ret)
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.pointers.len()
    }

    #[inline]
    pub fn element_len(&self, idx: usize) -> usize {
        if idx >= self.data.len() {
            return 0;
        }
        self.data[idx].len()
    }

    #[inline]
    pub fn lens(&self) -> &[u64] {
        self.lens.as_ref()
    }

    #[inline]
    pub fn pointers(&self) -> &[*const E] {
        &self.pointers[..]
    }
}
