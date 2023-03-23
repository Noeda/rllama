// This file contains platform-specific SIMD so that rest of rllama does not need to care which
// platform it is on.

use core::arch::aarch64::*;
use half::f16;

pub type I32x8 = int32x4x2_t;
pub type F32x8 = float32x4x2_t;
pub type I16x8 = int16x8_t;

/* ------------------ */
/* Loading and storing things */
/* ------------------ */

#[inline]
pub fn load_i16x8(ptr: *const I16x8) -> I16x8 {
    unsafe { vld1q_s16(ptr) }
}

#[inline]
pub fn store_i16x8(ptr: *mut I16x8, a: I16x8) {
    unsafe { vst1q_s16(ptr, a) }
}

#[inline]
pub fn load_f32x8(ptr: *const F32x8) -> F32x8 {
    unsafe { vld1q_f32_x2(ptr as *const f32) }
}

#[inline]
pub fn store_f32x8(ptr: *mut F32x8, a: F32x8) {
    unsafe { vst1q_f32_x2(ptr as *mut f32, a) }
}

#[inline]
pub fn gather_f32x8(ptr: *const f32, indices: I32x8) -> F32x8 {
    unsafe { _mm256_i32gather_ps(ptr, indices, 1) }
}

/* ------------------ */
/* Conversions        */
/* ------------------ */

#[inline]
pub fn i16x8_as_f16_to_f32x8(a: I16x8) -> F32x8 {
    unsafe { _mm256_cvtph_ps(a) }
}

#[inline]
pub fn f32x8_to_i16x8_as_f16(a: F32x8) -> I16x8 {
    unsafe { _mm256_cvtps_ph(a, 0) }
}

/*
 * Constants, creating from constants
 */

pub fn f32x8_zero() -> F32x8 {
    unsafe { _mm256_setzero_ps() }
}

pub fn i16x8_zero() -> I16x8 {
    unsafe { _mm_setzero_si128() }
}

pub fn f32x8_singleton(value: f32) -> F32x8 {
    unsafe { _mm256_set1_ps(value) }
}

pub fn i32x8_from_values(
    val0: i32,
    val1: i32,
    val2: i32,
    val3: i32,
    val4: i32,
    val5: i32,
    val6: i32,
    val7: i32,
) -> I32x8 {
    unsafe { _mm256_set_epi32(val0, val1, val2, val3, val4, val5, val6, val7) }
}

/*
 * Operations
 */

// FMA

// a * b + c
pub fn fma_f32x8(a: F32x8, b: F32x8, c: F32x8) -> F32x8 {
    unsafe { _mm256_fmadd_ps(a, b, c) }
}

// Horizontal sums

#[inline]
pub fn horizontal_sum_f32x8(mut ymm: __m256) -> f32 {
    unsafe {
        let ymm2 = _mm256_permute2f128_ps(ymm, ymm, 1);
        ymm = _mm256_add_ps(ymm, ymm2);
        ymm = _mm256_hadd_ps(ymm, ymm);
        ymm = _mm256_hadd_ps(ymm, ymm);
        _mm256_cvtss_f32(ymm)
    }
}

#[inline]
pub fn horizontal_sum_and_f32_to_f16(mut ymm: __m256) -> f16 {
    unsafe {
        let ymm2 = _mm256_permute2f128_ps(ymm, ymm, 1);
        ymm = _mm256_add_ps(ymm, ymm2);
        ymm = _mm256_hadd_ps(ymm, ymm);
        ymm = _mm256_hadd_ps(ymm, ymm);
        f16::from_f32(_mm256_cvtss_f32(ymm))
    }
}
