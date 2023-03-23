// This file contains platform-specific SIMD so that rest of rllama does not need to care which
// platform it is on.

use core::arch::x86_64::*;
use half::f16;
use std::fmt::Write;

pub type I32x8 = __m256i;
pub type F32x8 = __m256;
pub type I16x8 = __m128i;

/* ------------------ */
/* Loading and storing things */
/* ------------------ */

#[inline]
pub fn load_i16x8(ptr: *const I16x8) -> I16x8 {
    unsafe { _mm_loadu_si128(ptr) }
}

#[inline]
pub fn store_i16x8(ptr: *mut I16x8, a: I16x8) {
    unsafe { _mm_storeu_si128(ptr, a) }
}

#[inline]
pub fn load_f32x8(ptr: *const F32x8) -> F32x8 {
    unsafe { _mm256_loadu_ps(ptr as *const f32) }
}

#[inline]
pub fn store_f32x8(ptr: *mut F32x8, a: F32x8) {
    unsafe { _mm256_storeu_ps(ptr as *mut f32, a) }
}

#[inline]
pub fn gather_f32x8(ptr: *const f32, indices: I32x8) -> F32x8 {
    unsafe { _mm256_i32gather_ps(ptr, indices, 1) }
}

#[inline]
pub fn gather_scale4_f32x8(ptr: *const f32, indices: I32x8) -> F32x8 {
    unsafe { _mm256_i32gather_ps(ptr, indices, 4) }
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

#[inline]
/// Converts f32x8 to i32x8, by just casting the bits. I.e. it does not round any numbers or
/// anything, it just copies the bits.
pub fn f32x8_to_i32x8_bitcast(a: F32x8) -> I32x8 {
    unsafe { _mm256_castps_si256(a) }
}

/*
 * ------------------
 * Accessing individual elements
 */

// Rust has no const arguments (yet, maybe in future).
// So we have this awkward match statement in each of these.

#[inline]
pub fn f32x8_get(a: F32x8, idx: usize) -> f32 {
    unsafe {
        let a = f32x8_to_i32x8_bitcast(a);
        let a = match idx {
            0 => _mm256_extract_epi32(a, 0),
            1 => _mm256_extract_epi32(a, 1),
            2 => _mm256_extract_epi32(a, 2),
            3 => _mm256_extract_epi32(a, 3),
            4 => _mm256_extract_epi32(a, 4),
            5 => _mm256_extract_epi32(a, 5),
            6 => _mm256_extract_epi32(a, 6),
            7 => _mm256_extract_epi32(a, 7),
            _ => panic!("f32x8_get: index out of bounds"),
        };
        // bitcast the i32 back to f32
        core::mem::transmute(a)
    }
}

#[inline]
pub fn i32x8_get(a: I32x8, idx: usize) -> i32 {
    unsafe {
        let a = match idx {
            0 => _mm256_extract_epi32(a, 0),
            1 => _mm256_extract_epi32(a, 1),
            2 => _mm256_extract_epi32(a, 2),
            3 => _mm256_extract_epi32(a, 3),
            4 => _mm256_extract_epi32(a, 4),
            5 => _mm256_extract_epi32(a, 5),
            6 => _mm256_extract_epi32(a, 6),
            7 => _mm256_extract_epi32(a, 7),
            _ => panic!("i32x8_get: index out of bounds"),
        };
        a
    }
}

#[inline]
pub fn i16x8_get(a: I16x8, idx: usize) -> i16 {
    unsafe {
        let a = match idx {
            0 => _mm_extract_epi16(a, 0),
            1 => _mm_extract_epi16(a, 1),
            2 => _mm_extract_epi16(a, 2),
            3 => _mm_extract_epi16(a, 3),
            4 => _mm_extract_epi16(a, 4),
            5 => _mm_extract_epi16(a, 5),
            6 => _mm_extract_epi16(a, 6),
            7 => _mm_extract_epi16(a, 7),
            _ => panic!("i16x8_get: index out of bounds"),
        };
        a as i16
    }
}

/*
 * Constants, creating from constants
 */

#[inline]
pub fn f32x8_zero() -> F32x8 {
    unsafe { _mm256_setzero_ps() }
}

#[inline]
pub fn i16x8_zero() -> I16x8 {
    unsafe { _mm_setzero_si128() }
}

#[inline]
pub fn i16x8_singleton(value: i16) -> I16x8 {
    unsafe { _mm_set1_epi16(value) }
}

#[inline]
pub fn i16x8_singleton_u16(value: u16) -> I16x8 {
    unsafe { _mm_set1_epi16(value as i16) }
}

#[inline]
pub fn f32x8_singleton(value: f32) -> F32x8 {
    unsafe { _mm256_set1_ps(value) }
}

#[inline]
pub fn f32x8_from_values(
    val0: f32,
    val1: f32,
    val2: f32,
    val3: f32,
    val4: f32,
    val5: f32,
    val6: f32,
    val7: f32,
) -> F32x8 {
    unsafe { _mm256_set_ps(val0, val1, val2, val3, val4, val5, val6, val7) }
}

#[inline]
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

#[inline]
pub fn i32x8_from_values_u32(
    val0: u32,
    val1: u32,
    val2: u32,
    val3: u32,
    val4: u32,
    val5: u32,
    val6: u32,
    val7: u32,
) -> I32x8 {
    unsafe {
        _mm256_set_epi32(
            val0 as i32,
            val1 as i32,
            val2 as i32,
            val3 as i32,
            val4 as i32,
            val5 as i32,
            val6 as i32,
            val7 as i32,
        )
    }
}

#[inline]
pub fn i16x8_from_values(
    val0: i16,
    val1: i16,
    val2: i16,
    val3: i16,
    val4: i16,
    val5: i16,
    val6: i16,
    val7: i16,
) -> I16x8 {
    unsafe { _mm_set_epi16(val0, val1, val2, val3, val4, val5, val6, val7) }
}

/*
 * Operations
 */

// FMA

// a * b + c
#[inline]
pub fn fma_f32x8(a: F32x8, b: F32x8, c: F32x8) -> F32x8 {
    unsafe { _mm256_fmadd_ps(a, b, c) }
}

// bitwise and
#[inline]
pub fn and_i16x8(a: I16x8, b: I16x8) -> I16x8 {
    unsafe { _mm_and_si128(a, b) }
}

#[inline]
pub fn and_i32x8(a: I32x8, b: I32x8) -> I32x8 {
    unsafe { _mm256_and_si256(a, b) }
}

#[inline]
pub fn and_f32x8(a: F32x8, b: I32x8) -> F32x8 {
    unsafe { std::mem::transmute(_mm256_and_si256(std::mem::transmute(a), b)) }
}

// shift right by 4 bits exactly, for each individual i16 value.
// extends by zeros from left.
#[inline]
pub fn shift_right_by_4_i16x8(a: I16x8) -> I16x8 {
    unsafe { _mm_srli_epi16(a, 4) }
}

// shift right by half of an entire i16x8
// extends by zeros from left.
#[inline]
pub fn shift_right_by_64_i128(a: I16x8) -> I16x8 {
    unsafe { _mm_srli_si128(a, 64 / 8) }
}

// Extends 8 i8 values into 7 i16 values
//
// XXYYZZ -> 00XX00YY00ZZ
pub fn extend_i8_to_i16_i16x8(a: I16x8) -> I16x8 {
    unsafe { _mm_cvtepi8_epi16(a) }
}

// Extends 8 i8 values into 4 i32 values
pub fn extend_i8_to_i32_i32x8(a: I16x8) -> I32x8 {
    let i = extend_i8_to_i16_i16x8(a);
    unsafe { _mm256_cvtepu16_epi32(i) }
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

/*
 * Debugging
 */

/// Prints a binary representation of i16x8 to stdout in this form:
///
/// ```ignore
///       0                 0                 0               0
///     0x0000           0x0000           0x0000           0x0000
/// 0000000000000000 0000000000000000 0000000000000000 0000000000000000 etc.
/// ```
///
/// decimal on first line, hex on second, binary on third.
pub fn print_i16x8(a: I16x8) {
    let mut decimal_line = String::new();
    let mut hex_line = String::new();
    let mut binary_line = String::new();

    for i in 0..8 {
        let val = i16x8_get(a, i);
        write!(decimal_line, "{:>5}            ", val).unwrap();
        write!(hex_line, "0x{:04X}           ", val).unwrap();
        write!(binary_line, "{:016b} ", val).unwrap();
    }

    println!("{}", decimal_line.trim_end());
    println!("{}", hex_line.trim_end());
    println!("{}", binary_line.trim_end());
}

pub fn print_i32x8(a: I32x8) {
    let mut decimal_line = String::new();
    let mut hex_line = String::new();
    let mut binary_line = String::new();

    for i in 0..8 {
        let val = i32x8_get(a, i);
        write!(decimal_line, "{:>10}            ", val).unwrap();
        write!(hex_line, "0x{:08X}           ", val).unwrap();
        write!(binary_line, "{:032b} ", val).unwrap();
    }

    println!("{}", decimal_line.trim_end());
    println!("{}", hex_line.trim_end());
    println!("{}", binary_line.trim_end());
}
