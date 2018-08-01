use super::error::*;
use ffi::cudart::*;

use std::mem::size_of;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::os::raw::*;
use std::ptr::null_mut;
use std::slice::{from_raw_parts, from_raw_parts_mut};

#[derive(Debug)]
pub struct DVec<T> {
    ptr: *mut T,
    n: usize,
}

impl<T> DVec<T> {
    pub unsafe fn uninitialized(n: usize) -> Result<Self> {
        let mut ptr: *mut c_void = null_mut();
        cudaMalloc(&mut ptr as *mut *mut c_void, n * size_of::<T>()).check()?;
        Ok(DVec { ptr: ptr as *mut T, n })
    }

    pub fn fill_zero(&mut self) -> Result<()> {
        unsafe { cudaMemset(self.ptr as *mut c_void, 0, self.n * size_of::<T>()).check() }
    }

    pub fn new(n: usize) -> Result<Self> {
        let mut v = unsafe { Self::uninitialized(n) }?;
        v.fill_zero()?;
        Ok(v)
    }

    pub fn memcpy(&mut self, src: &[T]) {
        assert!(src.len() <= self.n);
        unsafe {
            cudaMemcpy(self.ptr as *mut c_void, src as *const _ as *const c_void, src.len(),
                    cudaMemcpyKind_cudaMemcpyHostToDevice);
        }
    }

}


impl<T> Drop for DVec<T> {
    fn drop(&mut self) {
        unsafe { cudaFree(self.ptr as *mut c_void) }
            .check()
            .expect("Free failed");
    }
}

#[cfg(test)]
mod tests {
    // TODO
}
