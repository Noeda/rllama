    fn matrix_mul_inplace_transposed_f32_and_k4bit(&mut self, src: &Tensor, other: &Tensor) {
        // Assume: size checks have been done already.
        assert!(src.dtype == TensorDType::Float32);
        assert!(other.dtype == TensorDType::K4BitQuantization);
        assert!(self.dtype == TensorDType::Float32);

        unsafe {
            let src_rows: usize = src.rows as usize;
            let src_cols: usize = src.cols as usize;
            let src_cols_capacity: usize = src.capacity_cols as usize;
            let other_cols: usize = other.cols as usize;
            let other_rows: usize = other.rows as usize;
            let other_cols_capacity: usize = other.capacity_cols as usize;
            let self_rows: usize = self.rows as usize;
            let self_cols: usize = self.cols as usize;
            let self_cols_capacity: usize = self.capacity_cols as usize;

            // src_cols_its == also the shared dimension between src and other.
            let src_cols_its = if src_cols % 32 == 0 {
                src_cols / 32
            } else {
                src_cols / 32 + 1
            };
            debug_assert!(!other.q4_data.is_null());

            let src_data_wrap: WrappedPtr = WrappedPtr::wrap(src.data);
            let other_data: WrappedPtr = WrappedPtr::wrap(other.data);
            let tgt_data: WrappedPtr = WrappedPtr::wrap(self.data);
            let other_q4_data: WrappedPtr = WrappedPtr::wrap(other.q4_data);

            let nthreads: usize = rayon::current_num_threads();
            (0..nthreads).into_par_iter().for_each(|thread_idx| {
                let other_q4_data: *const u8 = other_q4_data.unwrap() as *const u8;
                let src_data: *const f32 = src_data_wrap.unwrap() as *const f32;
                let other_data: *const u8 = other_data.unwrap() as *const u8;
                let tgt_data: *mut f32 = tgt_data.unwrap() as *mut f32;

                for row in 0..self_rows {
                    for col in 0..self_cols {
                        let row_col = row * self_cols + col;
                        if row_col % nthreads != thread_idx {
                            continue;
                        }

                        let quant0 = load_i16x8(other_q4_data.add(col * 32) as *const I16x8);
                        let quant1 = load_i16x8(other_q4_data.add(col * 32 + 16) as *const I16x8);
                        let quants: [F32x8; 2] =
                            [i16x8_as_f16_to_f32x8(quant0), i16x8_as_f16_to_f32x8(quant1)];

                        #[inline]
                        fn load_f32(
                            src: *const f32,
                            row: usize,
                            col: usize,
                            ncols: usize,
                            nrows: usize,
                            cols_capacity: usize,
                        ) -> F32x8 {
                            unsafe {
                                if row >= nrows || col >= ncols {
                                    f32x8_zero()
                                } else {
                                    load_f32x8(src.add(row * cols_capacity + col) as *const F32x8)
                                }
                            }
                        }

                        #[inline]
                        fn load_k4_to_f32(
                            tensor: &Tensor,
                            row: usize,
                            col: usize,
                            nrows: usize,
                            quants: *const F32x8,
                        ) -> (F32x8, F32x8, F32x8, F32x8) {
                            unsafe {
                                let m: u32 = 0xFFFFFFFF;
                                let masks: [I32x8; 8] = [
                                    i32x8_from_values_u32(m, m, m, m, m, m, m, m),
                                    i32x8_from_values_u32(0, m, m, m, m, m, m, m),
                                    i32x8_from_values_u32(0, 0, m, m, m, m, m, m),
                                    i32x8_from_values_u32(0, 0, 0, m, m, m, m, m),
                                    i32x8_from_values_u32(0, 0, 0, 0, m, m, m, m),
                                    i32x8_from_values_u32(0, 0, 0, 0, 0, m, m, m),
                                    i32x8_from_values_u32(0, 0, 0, 0, 0, 0, m, m),
                                    i32x8_from_values_u32(0, 0, 0, 0, 0, 0, 0, m),
                                ];
                                let nomask: I32x8 = i32x8_from_values_u32(m, m, m, m, m, m, m, m);
                                let fullmask: I32x8 = i32x8_from_values_u32(0, 0, 0, 0, 0, 0, 0, 0);

                                if row < nrows {
                                    let col = col as i64;
                                    let ncols = tensor.cols;
                                    let (addr, side) = tensor.q4_address(row as i64, col);
                                    let i = load_i16x8(addr as *const I16x8);
                                    let even_mask = i16x8_singleton_u16(0x0F0F);
                                    let odd_mask = i16x8_singleton_u16(0xF0F0);
                                    let evens = and_i16x8(i, even_mask);
                                    let odds = and_i16x8(i, odd_mask);
                                    let odds = shift_right_by_4_i16x8(odds);

                                    let indices1 = extend_i8_to_i32_i32x8(odds);
                                    let odds_shifted = shift_right_by_64_i128(odds);
                                    let indices2 = extend_i8_to_i32_i32x8(odds_shifted);
                                    let indices3 = extend_i8_to_i32_i32x8(evens);
                                    let indices4 =
                                        extend_i8_to_i32_i32x8(shift_right_by_64_i128(evens));

                                    let unquantized1: F32x8 =
                                        gather_scale4_f32x8(quants as *const f32, indices1);
                                    let unquantized2: F32x8 =
                                        gather_scale4_f32x8(quants as *const f32, indices2);
                                    let unquantized3: F32x8 =
                                        gather_scale4_f32x8(quants as *const f32, indices3);
                                    let unquantized4: F32x8 =
                                        gather_scale4_f32x8(quants as *const f32, indices4);
                                    let quan1_mask: I32x8 = if col <= ncols - 8 {
                                        nomask
                                    } else if col < ncols {
                                        masks[(col % 8) as usize]
                                    } else {
                                        fullmask
                                    };
                                    let quan2_mask: I32x8 = if col <= ncols - 16 {
                                        nomask
                                    } else if col < ncols - 8 {
                                        masks[(col % 8) as usize]
                                    } else {
                                        fullmask
                                    };
                                    let quan3_mask: I32x8 = if col <= ncols - 24 {
                                        nomask
                                    } else if col < ncols - 16 {
                                        masks[(col % 8) as usize]
                                    } else {
                                        fullmask
                                    };
                                    let quan4_mask: I32x8 = if col <= ncols - 32 {
                                        nomask
                                    } else if col < ncols - 24 {
                                        masks[(col % 8) as usize]
                                    } else {
                                        fullmask
                                    };
                                    let unquantized1 = and_f32x8(unquantized1, quan1_mask);
                                    let unquantized2 = and_f32x8(unquantized2, quan2_mask);
                                    let unquantized3 = and_f32x8(unquantized3, quan3_mask);
                                    let unquantized4 = and_f32x8(unquantized4, quan4_mask);
                                    (unquantized1, unquantized2, unquantized3, unquantized4)
                                } else {
                                    (f32x8_zero(), f32x8_zero(), f32x8_zero(), f32x8_zero())
                                }
                            }
                        }

                        let mut targets8: [F32x8; 4] =
                            [f32x8_zero(), f32x8_zero(), f32x8_zero(), f32x8_zero()];

                        for p in 0..src_cols_its {
                            let (other8_0, other8_1, other8_2, other8_3) =
                                load_k4_to_f32(&other, col, p * 32, other_rows, quants.as_ptr());
                            let src8_0 = load_f32(
                                src_data,
                                row,
                                p * 32,
                                src_cols,
                                src_rows,
                                src_cols_capacity,
                            );
                            let src8_1 = load_f32(
                                src_data,
                                row,
                                p * 32 + 8,
                                src_cols,
                                src_rows,
                                src_cols_capacity,
                            );
                            let src8_2 = load_f32(
                                src_data,
                                row,
                                p * 32 + 16,
                                src_cols,
                                src_rows,
                                src_cols_capacity,
                            );
                            let src8_3 = load_f32(
                                src_data,
                                row,
                                p * 32 + 24,
                                src_cols,
                                src_rows,
                                src_cols_capacity,
                            );
                            targets8[0] = fma_f32x8(src8_0, other8_0, targets8[0]);
                            targets8[1] = fma_f32x8(src8_1, other8_1, targets8[1]);
                            targets8[2] = fma_f32x8(src8_2, other8_2, targets8[2]);
                            targets8[3] = fma_f32x8(src8_3, other8_3, targets8[3]);
                        }
                        let target0 = horizontal_sum_f32x8(targets8[0]);
                        let target1 = horizontal_sum_f32x8(targets8[1]);
                        let target2 = horizontal_sum_f32x8(targets8[2]);
                        let target3 = horizontal_sum_f32x8(targets8[3]);
                        let target = target0 + target1 + target2 + target3;
                        *tgt_data.add(row * self_cols_capacity + col) = target;
                    }
                }
            });
        }
    }

    fn matrix_mul_inplace_transposed_k4bit_and_f32(&mut self, src: &Tensor, other: &Tensor) {
        // Assume: size checks have been done already.
        assert!(src.dtype == TensorDType::K4BitQuantization);
        assert!(other.dtype == TensorDType::Float32);
        assert!(self.dtype == TensorDType::Float32);

        unsafe {
            let src_rows: usize = src.rows as usize;
            let src_cols: usize = src.cols as usize;
            let src_cols_capacity: usize = src.capacity_cols as usize;
            let other_cols: usize = other.cols as usize;
            let other_rows: usize = other.rows as usize;
            let other_cols_capacity: usize = other.capacity_cols as usize;
            let self_rows: usize = self.rows as usize;
            let self_cols: usize = self.cols as usize;
            let self_cols_capacity: usize = self.capacity_cols as usize;

            // src_cols_its == also the shared dimension between src and other.
            let src_cols_its = if src_cols % 32 == 0 {
                src_cols / 32
            } else {
                src_cols / 32 + 1
            };
            debug_assert!(!src.q4_data.is_null());

            let src_data_wrap: WrappedPtr = WrappedPtr::wrap(src.data);
            let other_data: WrappedPtr = WrappedPtr::wrap(other.data);
            let tgt_data: WrappedPtr = WrappedPtr::wrap(self.data);
            let src_q4_data: WrappedPtr = WrappedPtr::wrap(src.q4_data);

            let nthreads: usize = rayon::current_num_threads();
            (0..nthreads).into_par_iter().for_each(|thread_idx| {
                let src_q4_data: *const u8 = src_q4_data.unwrap() as *const u8;
                let src_data: *const u8 = src_data_wrap.unwrap() as *const u8;
                let other_data: *const f32 = other_data.unwrap() as *const f32;
                let tgt_data: *mut f32 = tgt_data.unwrap() as *mut f32;

                for row in 0..self_rows {
                    let quant0 = load_i16x8(src_q4_data.add(row * 32) as *const I16x8);
                    let quant1 = load_i16x8(src_q4_data.add(row * 32 + 16) as *const I16x8);
                    let quants: [F32x8; 2] =
                        [i16x8_as_f16_to_f32x8(quant0), i16x8_as_f16_to_f32x8(quant1)];

                    for col in 0..self_cols {
                        let row_col = row * self_cols + col;
                        if row_col % nthreads != thread_idx {
                            continue;
                        }

                        #[inline]
                        fn load_f32(
                            other: *const f32,
                            row: usize,
                            col: usize,
                            ncols: usize,
                            nrows: usize,
                            cols_capacity: usize,
                        ) -> F32x8 {
                            unsafe {
                                if row >= nrows || col >= ncols {
                                    f32x8_zero()
                                } else {
                                    load_f32x8(other.add(row * cols_capacity + col) as *const F32x8)
                                }
                            }
                        }

                        #[inline]
                        fn load_k4_to_f32(
                            tensor: &Tensor,
                            row: usize,
                            col: usize,
                            nrows: usize,
                            quants: *const F32x8,
                        ) -> (F32x8, F32x8, F32x8, F32x8) {
                            unsafe {
                                let m: u32 = 0xFFFFFFFF;
                                let masks: [I32x8; 8] = [
                                    i32x8_from_values_u32(m, m, m, m, m, m, m, m),
                                    i32x8_from_values_u32(0, m, m, m, m, m, m, m),
                                    i32x8_from_values_u32(0, 0, m, m, m, m, m, m),
                                    i32x8_from_values_u32(0, 0, 0, m, m, m, m, m),
                                    i32x8_from_values_u32(0, 0, 0, 0, m, m, m, m),
                                    i32x8_from_values_u32(0, 0, 0, 0, 0, m, m, m),
                                    i32x8_from_values_u32(0, 0, 0, 0, 0, 0, m, m),
                                    i32x8_from_values_u32(0, 0, 0, 0, 0, 0, 0, m),
                                ];
                                let nomask: I32x8 = i32x8_from_values_u32(m, m, m, m, m, m, m, m);
                                let fullmask: I32x8 = i32x8_from_values_u32(0, 0, 0, 0, 0, 0, 0, 0);

                                if row < nrows {
                                    let col = col as i64;
                                    let ncols = tensor.cols;
                                    let (addr, side) = tensor.q4_address(row as i64, col);
                                    let i = load_i16x8(addr as *const I16x8);
                                    let even_mask = i16x8_singleton_u16(0x0F0F);
                                    let odd_mask = i16x8_singleton_u16(0xF0F0);
                                    let evens = and_i16x8(i, even_mask);
                                    let odds = and_i16x8(i, odd_mask);
                                    let odds = shift_right_by_4_i16x8(odds);

                                    let indices1 = extend_i8_to_i32_i32x8(odds);
                                    let odds_shifted = shift_right_by_64_i128(odds);
                                    let indices2 = extend_i8_to_i32_i32x8(odds_shifted);
                                    let indices3 = extend_i8_to_i32_i32x8(evens);
                                    let indices4 =
                                        extend_i8_to_i32_i32x8(shift_right_by_64_i128(evens));

                                    let unquantized1: F32x8 =
                                        gather_scale4_f32x8(quants as *const f32, indices1);
                                    let unquantized2: F32x8 =
                                        gather_scale4_f32x8(quants as *const f32, indices2);
                                    let unquantized3: F32x8 =
                                        gather_scale4_f32x8(quants as *const f32, indices3);
                                    let unquantized4: F32x8 =
                                        gather_scale4_f32x8(quants as *const f32, indices4);
                                    let quan1_mask: I32x8 = if col <= ncols - 8 {
                                        nomask
                                    } else if col < ncols {
                                        masks[(col % 8) as usize]
                                    } else {
                                        fullmask
                                    };
                                    let quan2_mask: I32x8 = if col <= ncols - 16 {
                                        nomask
                                    } else if col < ncols - 8 {
                                        masks[(col % 8) as usize]
                                    } else {
                                        fullmask
                                    };
                                    let quan3_mask: I32x8 = if col <= ncols - 24 {
                                        nomask
                                    } else if col < ncols - 16 {
                                        masks[(col % 8) as usize]
                                    } else {
                                        fullmask
                                    };
                                    let quan4_mask: I32x8 = if col <= ncols - 32 {
                                        nomask
                                    } else if col < ncols - 24 {
                                        masks[(col % 8) as usize]
                                    } else {
                                        fullmask
                                    };
                                    let unquantized1 = and_f32x8(unquantized1, quan1_mask);
                                    let unquantized2 = and_f32x8(unquantized2, quan2_mask);
                                    let unquantized3 = and_f32x8(unquantized3, quan3_mask);
                                    let unquantized4 = and_f32x8(unquantized4, quan4_mask);
                                    (unquantized1, unquantized2, unquantized3, unquantized4)
                                } else {
                                    (f32x8_zero(), f32x8_zero(), f32x8_zero(), f32x8_zero())
                                }
                            }
                        }

                        let mut targets8: [F32x8; 4] =
                            [f32x8_zero(), f32x8_zero(), f32x8_zero(), f32x8_zero()];

                        for p in 0..src_cols_its {
                            let other8_0: F32x8 = load_f32(
                                other_data,
                                col,
                                p * 32,
                                other_cols,
                                other_rows,
                                other_cols_capacity,
                            );
                            let other8_1: F32x8 = load_f32(
                                other_data,
                                col,
                                p * 32 + 8,
                                other_cols,
                                other_rows,
                                other_cols_capacity,
                            );
                            let other8_2: F32x8 = load_f32(
                                other_data,
                                col,
                                p * 32 + 16,
                                other_cols,
                                other_rows,
                                other_cols_capacity,
                            );
                            let other8_3: F32x8 = load_f32(
                                other_data,
                                col,
                                p * 32 + 24,
                                other_cols,
                                other_rows,
                                other_cols_capacity,
                            );
                            let (src8_0, src8_1, src8_2, src8_3): (F32x8, F32x8, F32x8, F32x8) =
                                load_k4_to_f32(&src, row, p * 32, src_rows, quants.as_ptr());
                            targets8[0] = fma_f32x8(src8_0, other8_0, targets8[0]);
                            targets8[1] = fma_f32x8(src8_1, other8_1, targets8[1]);
                            targets8[2] = fma_f32x8(src8_2, other8_2, targets8[2]);
                            targets8[3] = fma_f32x8(src8_3, other8_3, targets8[3]);
                        }
                        let target0 = horizontal_sum_f32x8(targets8[0]);
                        let target1 = horizontal_sum_f32x8(targets8[1]);
                        let target2 = horizontal_sum_f32x8(targets8[2]);
                        let target3 = horizontal_sum_f32x8(targets8[3]);
                        let target = target0 + target1 + target2 + target3;
                        *tgt_data.add(row * self_cols_capacity + col) = target;
                    }
                }
            });
        }
    }
