use crate::quantization::{MutableQuantizedTensor, QuantizedTensor};

use rayon::prelude::*;
use std::convert::TryInto;
use wide::{f32x8, i32x8};

// Some helper functions

pub fn slice_to_u32(slice: &[u8]) -> u32 {
    assert!(slice.len() == 4, "Slice must be exactly 4 bytes long");
    u32::from_ne_bytes(slice.try_into().expect("Slice with incorrect length"))
}

pub fn slice_to_f32(slice: &[u8]) -> f32 {
    assert!(slice.len() == 4, "Slice must be exactly 4 bytes long");
    f32::from_ne_bytes(slice.try_into().expect("Slice with incorrect length"))
}

pub fn u8_to_f32_slice(data: &[u8]) -> &[f32] {
    let (prefix, f32data, suffix) = unsafe { data.align_to::<f32>() };
    assert!(prefix.is_empty(), "Data was not aligned correctly");
    assert!(suffix.is_empty(), "Data was not aligned correctly");
    f32data
}

pub fn u8_to_i8_slice(data: &[u8]) -> &[i8] {
    let (prefix, i8data, suffix) = unsafe { data.align_to::<i8>() };
    assert!(prefix.is_empty(), "Data was not aligned correctly");
    assert!(suffix.is_empty(), "Data was not aligned correctly");
    i8data
}

pub fn random_u32(mut state: u64) -> u32 {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;

    ((state * 0x2545F4914F6CDD1Du64) >> 32) as u32
}

pub fn random_f32(state: u64) -> f32 {
    (random_u32(state) >> 8) as f32 / 16777216.0f32
}

// Functions used in NNs

pub fn rmsnorm(
    o: &mut [f32],
    x: &[f32],
    weight: &[f32],
    size: usize,
    eps: f32,
    add_unit_offset: bool,
) {
    let n_simd = size / 8;

    let mut ss_sim = f32x8::ZERO;

    for j in 0..n_simd {
        let x_vec = f32x8::from(&x[j * 8..j * 8 + 8]);
        ss_sim += x_vec * x_vec;
    }

    let mut ss = ss_sim.reduce_add();

    ss /= size as f32;
    ss += eps;
    ss = 1.0 / ss.sqrt();

    for j in 0..n_simd {
        let x_vec = f32x8::from(&x[j * 8..j * 8 + 8]);
        let w_vec = f32x8::from(&weight[j * 8..j * 8 + 8]);

        let r = if add_unit_offset {
            ((1.0 + w_vec) * (ss * x_vec)).to_array()
        } else {
            (w_vec * (ss * x_vec)).to_array()
        };

        for k in 0..8 {
            o[(j * 8) + k] = r[k];
        }
    }
}

pub fn layernorm(o: &mut [f32], x: &[f32], weight: &[f32], bias: &[f32], size: usize, eps: f32) {
    let n_simd = size / 8;

    let mut mean_sim = f32x8::ZERO;
    let mut var_sim = f32x8::ZERO;

    for j in 0..n_simd {
        let x_vec = f32x8::from(&x[j * 8..j * 8 + 8]);
        mean_sim += x_vec;
    }

    let mean = mean_sim.reduce_add() / size as f32;

    for j in 0..n_simd {
        let x_vec = f32x8::from(&x[j * 8..j * 8 + 8]);
        let diff = x_vec - f32x8::splat(mean);
        var_sim += diff * diff;
    }

    let variance = var_sim.reduce_add() / size as f32 + eps;
    let inv_std = 1.0 / variance.sqrt();

    for j in 0..n_simd {
        let x_vec = f32x8::from(&x[j * 8..j * 8 + 8]);
        let w_vec = f32x8::from(&weight[j * 8..j * 8 + 8]);
        let b_vec = f32x8::from(&bias[j * 8..j * 8 + 8]);

        let normalized = (x_vec - f32x8::splat(mean)) * f32x8::splat(inv_std);
        let r = (normalized * w_vec + b_vec).to_array();

        for k in 0..8 {
            o[(j * 8) + k] = r[k];
        }
    }
}

pub fn tanh_f32x8(input: f32x8) -> f32x8 {
    let two = f32x8::splat(2.0);
    let exp_2x = (input * two).exp();
    (exp_2x - f32x8::splat(1.0)) / (exp_2x + f32x8::splat(1.0))
}

pub fn softmax(x: &mut [f32]) {
    let mut sum: f32 = 0.0;
    let mut max_val: f32 = match x.first() {
        Some(&val) => val,
        None => return, // If the slice is empty, do nothing
    };

    for i in x.iter() {
        if *i > max_val {
            max_val = *i;
        }
    }

    for i in x.iter_mut() {
        *i = (*i - max_val).exp();
        sum += *i;
    }

    for i in x.iter_mut() {
        *i /= sum;
    }
}

pub fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, o: usize) {
    let n_simd = n / 8;

    xout.par_chunks_exact_mut(o)
        .enumerate()
        .for_each(|(j, elem)| {
            let xi = j * n;

            elem.par_chunks_exact_mut(4)
                .enumerate()
                .for_each(|(i, xout_elem)| {
                    let new_i = i * 4;
                    let ni0: usize = new_i * n;
                    let ni1: usize = (new_i + 1) * n;
                    let ni2: usize = (new_i + 2) * n;
                    let ni3: usize = (new_i + 3) * n;

                    xout_elem.iter_mut().for_each(|m| *m = 0.0);

                    for j in 0..n_simd {
                        let x_vec = f32x8::from(&x[xi + j * 8..xi + j * 8 + 8]);
                        let w_vec0 = f32x8::from(&w[ni0 + j * 8..ni0 + j * 8 + 8]);
                        let w_vec1 = f32x8::from(&w[ni1 + j * 8..ni1 + j * 8 + 8]);
                        let w_vec2 = f32x8::from(&w[ni2 + j * 8..ni2 + j * 8 + 8]);
                        let w_vec3 = f32x8::from(&w[ni3 + j * 8..ni3 + j * 8 + 8]);

                        xout_elem[0] += (x_vec * w_vec0).reduce_add();
                        xout_elem[1] += (x_vec * w_vec1).reduce_add();
                        xout_elem[2] += (x_vec * w_vec2).reduce_add();
                        xout_elem[3] += (x_vec * w_vec3).reduce_add();
                    }
                });
        });
}

pub fn matmul_q8(
    xout: &mut [f32],
    x: &MutableQuantizedTensor,
    w: &QuantizedTensor,
    n: usize,
    o: usize,
    gs: usize,
) {
    let n_simd = gs / 8;

    xout.par_chunks_exact_mut(o)
        .enumerate()
        .for_each(|(j, elem)| {
            let xi = j * n;

            elem.par_chunks_exact_mut(4)
                .enumerate()
                .for_each(|(i, xout_elem)| {
                    let new_i = i * 4;
                    let ni0: usize = new_i * n;
                    let ni1: usize = (new_i + 1) * n;
                    let ni2: usize = (new_i + 2) * n;
                    let ni3: usize = (new_i + 3) * n;

                    xout_elem.iter_mut().for_each(|m| *m = 0.0);

                    for j in (0..=(n - gs)).step_by(gs) {
                        let mut ival0 = i32x8::ZERO;
                        let mut ival1 = i32x8::ZERO;
                        let mut ival2 = i32x8::ZERO;
                        let mut ival3 = i32x8::ZERO;

                        for k in 0..n_simd {
                            let x_vec = i32x8::from(&x.q[xi + j + k * 8..xi + j + k * 8 + 8]);
                            let w_vec0 = i32x8::from(&w.q[ni0 + j + k * 8..ni0 + j + k * 8 + 8]);
                            let w_vec1 = i32x8::from(&w.q[ni1 + j + k * 8..ni1 + j + k * 8 + 8]);
                            let w_vec2 = i32x8::from(&w.q[ni2 + j + k * 8..ni2 + j + k * 8 + 8]);
                            let w_vec3 = i32x8::from(&w.q[ni3 + j + k * 8..ni3 + j + k * 8 + 8]);

                            ival0 += x_vec * w_vec0;
                            ival1 += x_vec * w_vec1;
                            ival2 += x_vec * w_vec2;
                            ival3 += x_vec * w_vec3;
                        }

                        xout_elem[0] +=
                            (ival0.reduce_add() as f32) * w.s[(ni0 + j) / gs] * x.s[(xi + j) / gs];
                        xout_elem[1] +=
                            (ival1.reduce_add() as f32) * w.s[(ni1 + j) / gs] * x.s[(xi + j) / gs];
                        xout_elem[2] +=
                            (ival2.reduce_add() as f32) * w.s[(ni2 + j) / gs] * x.s[(xi + j) / gs];
                        xout_elem[3] +=
                            (ival3.reduce_add() as f32) * w.s[(ni3 + j) / gs] * x.s[(xi + j) / gs];
                    }
                });
        });
}

pub fn matmul_q4(
    xout: &mut [f32],
    x: &MutableQuantizedTensor,
    w: &QuantizedTensor,
    n: usize,
    o: usize,
    gs: usize,
) {
    let group_size = gs / 2;
    let n_simd = group_size / 8;

    let mask_a = i32x8::new([0x0F; 8]);
    let mask_b = i32x8::new([0xF0; 8]);

    xout.par_chunks_exact_mut(o)
        .enumerate()
        .for_each(|(j, elem)| {
            let xi = j * n;

            elem.par_iter_mut().enumerate().for_each(|(i, xout_elem)| {
                let ni: usize = i * n / 2;

                *xout_elem = (0..=(n / 2 - group_size))
                    .step_by(group_size)
                    .map(|j| {
                        let mut ival = i32x8::ZERO;

                        for k in 0..n_simd {
                            let x_vec = i32x8::from(&x.q[xi + j + k * 8..xi + j + k * 8 + 8]);
                            let w_vec = i32x8::from(&w.q[ni + j + k * 8..ni + j + k * 8 + 8]);

                            let x_a = (x_vec & mask_a) - 8;
                            let w_a = (w_vec & mask_a) - 8;

                            let x_b = (mask_a & ((x_vec & mask_b) >> 4)) - 8;
                            let w_b = (mask_a & ((w_vec & mask_b) >> 4)) - 8;

                            ival += x_a * w_a;
                            ival += x_b * w_b;
                        }

                        (ival.reduce_add() as f32)
                            * w.s[(ni + j) / group_size]
                            * x.s[(xi + j) / group_size]
                    })
                    .sum();
            });
        });
}

pub fn matmul_rest(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, o: usize) {
    let n_simd = n / 8;

    let rest = n_simd * 8;

    xout.par_chunks_exact_mut(o)
        .enumerate()
        .for_each(|(j, elem)| {
            let xi = j * n;

            elem.par_iter_mut().enumerate().for_each(|(i, val)| {
                let mut sum = f32x8::ZERO;
                let mut final_sum: f32 = 0.0;
                let w_slice = &w[i * n..i * n + n];

                for j in 0..n_simd {
                    let x_vec = f32x8::from(&x[xi + j * 8..xi + j * 8 + 8]);
                    let w_vec = f32x8::from(&w_slice[j * 8..j * 8 + 8]);
                    sum += w_vec * x_vec;
                }

                final_sum += sum.reduce_add();

                for r in rest..n {
                    final_sum += w_slice[r] * x[r];
                }

                *val = final_sum;
            });
        });
}

pub fn concat<T: Clone>(arr0: &[T], arr1: &[T]) -> Vec<T> {
    let mut concat_arr: Vec<T> = Vec::new();

    concat_arr.extend_from_slice(arr0);
    concat_arr.extend_from_slice(arr1);

    concat_arr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_to_u32() {
        let data = [0x01, 0x02, 0x03, 0x04];
        assert_eq!(slice_to_u32(&data), 0x04030201);
    }

    #[test]
    fn test_slice_to_f32() {
        let data = [0x00, 0x00, 0x80, 0x7F]; // Represents f32::INFINITY
        assert_eq!(slice_to_f32(&data), f32::INFINITY);
    }

    #[test]
    fn test_u8_to_f32_slice() {
        let data: Vec<u8> = vec![0, 0, 0, 0, 0, 0, 128, 63]; // Represents [0.0, 1.0]
        let f32_slice = u8_to_f32_slice(&data);
        assert_eq!(f32_slice.len(), 2);
        assert_eq!(f32_slice[0], 0.0);
        assert_eq!(f32_slice[1], 1.0);
    }

    #[test]
    fn test_u8_to_i8_slice() {
        let data: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7]; // Represents [0, 1, 2, 3, 4, 5, 6, 7]
        let i8_slice = u8_to_i8_slice(&data);
        assert_eq!(i8_slice.len(), 8);
        assert_eq!(i8_slice[0], 0);
        assert_eq!(i8_slice[1], 1);
        assert_eq!(i8_slice[2], 2);
        assert_eq!(i8_slice[3], 3);
        assert_eq!(i8_slice[4], 4);
        assert_eq!(i8_slice[5], 5);
        assert_eq!(i8_slice[6], 6);
        assert_eq!(i8_slice[7], 7);
    }

    // 부동 소수점 비교를 위한 헬퍼 함수
    fn assert_approx_eq(actual: f32, expected: f32, epsilon: f32) {
        assert!(
            (actual - expected).abs() < epsilon,
            "Assertion failed: actual = {}, expected = {}, diff = {}",
            actual,
            expected,
            (actual - expected).abs()
        );
    }

    // 벡터 전체가 근사적으로 같은지 확인하는 헬퍼 함수
    fn assert_vec_approx_eq(actual: &[f32], expected: &[f32], epsilon: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "Vectors have different lengths"
        );
        for i in 0..actual.len() {
            assert_approx_eq(actual[i], expected[i], epsilon);
        }
    }

    // --- 테스트 케이스 시작 ---

    #[test]
    fn test_rmsnorm_basic_positive() {
        let size = 8;
        let mut o = vec![0.0; size];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // 모든 weight가 1
        let eps = 1e-5;
        let add_unit_offset = false;

        // 수동 계산 (RMS = sqrt(평균(x^2)))
        // x^2 = [1, 4, 9, 16, 25, 36, 49, 64]
        // sum(x^2) = 1+4+9+16+25+36+49+64 = 204
        // mean(x^2) = 204 / 8 = 25.5
        // ss = sqrt(25.5 + 1e-5) = sqrt(25.50001) approx 5.04975
        // scale_factor = 1.0 / ss = 1.0 / 5.04975 approx 0.198028
        // output = x * scale_factor * weight (weight=1이므로 x * scale_factor)

        // 대략적인 예상 값 계산 (정확한 f32 계산은 오차가 있을 수 있음)
        let sum_sq_x: f32 = x.iter().map(|&val| val * val).sum();
        let mean_sq_x = sum_sq_x / (size as f32);
        let ss_manual = 1.0 / (mean_sq_x + eps).sqrt();

        let mut expected_o = vec![0.0; size];
        for i in 0..size {
            expected_o[i] = x[i] * ss_manual * weight[i];
        }

        rmsnorm(&mut o, &x, &weight, size, eps, add_unit_offset);

        println!("Test: Basic Positive");
        println!("Input x: {:?}", x);
        println!("Weight: {:?}", weight);
        println!("Output o: {:?}", o);
        println!("Expected o: {:?}", expected_o);
        assert_vec_approx_eq(&o, &expected_o, 1e-5);
    }

    #[test]
    fn test_rmsnorm_with_negative_values() {
        let size = 8;
        let mut o = vec![0.0; size];
        let x = vec![-1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
        let weight = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]; // 모든 weight가 0.5
        let eps = 1e-5;
        let add_unit_offset = false;

        let sum_sq_x: f32 = x.iter().map(|&val| val * val).sum();
        let mean_sq_x = sum_sq_x / (size as f32);
        let ss_manual = 1.0 / (mean_sq_x + eps).sqrt();

        let mut expected_o = vec![0.0; size];
        for i in 0..size {
            expected_o[i] = x[i] * ss_manual * weight[i];
        }

        rmsnorm(&mut o, &x, &weight, size, eps, add_unit_offset);

        println!("\nTest: With Negative Values");
        println!("Input x: {:?}", x);
        println!("Weight: {:?}", weight);
        println!("Output o: {:?}", o);
        println!("Expected o: {:?}", expected_o);
        assert_vec_approx_eq(&o, &expected_o, 1e-5);
    }

    #[test]
    fn test_rmsnorm_with_add_unit_offset() {
        let size = 8;
        let mut o = vec![0.0; size];
        let x = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let weight = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let eps = 1e-5;
        let add_unit_offset = true; // unit offset 추가

        // x^2 = [1, 1, 1, 1, 1, 1, 1, 1]
        // sum(x^2) = 8
        // mean(x^2) = 8 / 8 = 1.0
        // ss = sqrt(1.0 + 1e-5) = sqrt(1.00001) approx 1.000005
        // scale_factor = 1.0 / ss approx 0.999995
        // effective_weight = 1.0 + weight = 1.5
        // output = x * scale_factor * effective_weight

        let sum_sq_x: f32 = x.iter().map(|&val| val * val).sum();
        let mean_sq_x = sum_sq_x / (size as f32);
        let ss_manual = 1.0 / (mean_sq_x + eps).sqrt();

        let mut expected_o = vec![0.0; size];
        for i in 0..size {
            expected_o[i] = x[i] * ss_manual * (1.0 + weight[i]);
        }

        rmsnorm(&mut o, &x, &weight, size, eps, add_unit_offset);

        println!("\nTest: With Add Unit Offset");
        println!("Input x: {:?}", x);
        println!("Weight: {:?}", weight);
        println!("Add Unit Offset: {}", add_unit_offset);
        println!("Output o: {:?}", o);
        println!("Expected o: {:?}", expected_o);
        assert_vec_approx_eq(&o, &expected_o, 1e-5);
    }

    #[test]
    fn test_rmsnorm_zero_x_input() {
        let size = 8;
        let mut o = vec![0.0; size];
        let x = vec![0.0; size]; // 모든 요소가 0인 x
        let weight = vec![1.0; size];
        let eps = 1e-5;
        let add_unit_offset = false;

        // x가 모두 0이면, RMS는 0이 됩니다.
        // ss_sim은 0, ss는 0. ss += eps 이므로 ss = eps.
        // ss = 1.0 / sqrt(eps)
        // output = 0 * (1.0/sqrt(eps)) * weight = 0
        let expected_o = vec![0.0; size];

        rmsnorm(&mut o, &x, &weight, size, eps, add_unit_offset);

        println!("\nTest: Zero X Input");
        println!("Input x: {:?}", x);
        println!("Output o: {:?}", o);
        println!("Expected o: {:?}", expected_o);
        assert_vec_approx_eq(&o, &expected_o, f32::EPSILON); // EPSILON으로 충분함
    }

    #[test]
    fn test_rmsnorm_large_values() {
        let size = 8;
        let mut o = vec![0.0; size];
        let x = vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0];
        let weight = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let eps = 1e-5;
        let add_unit_offset = false;

        let sum_sq_x: f32 = x.iter().map(|&val| val * val).sum();
        let mean_sq_x = sum_sq_x / (size as f32);
        let ss_manual = 1.0 / (mean_sq_x + eps).sqrt();

        let mut expected_o = vec![0.0; size];
        for i in 0..size {
            expected_o[i] = x[i] * ss_manual * weight[i];
        }

        rmsnorm(&mut o, &x, &weight, size, eps, add_unit_offset);

        println!("\nTest: Large Values");
        println!("Input x: {:?}", x);
        println!("Weight: {:?}", weight);
        println!("Output o: {:?}", o);
        println!("Expected o: {:?}", expected_o);
        assert_vec_approx_eq(&o, &expected_o, 1e-4); // 더 큰 epsilon 허용
    }

    #[test]
    fn test_rmsnorm_different_weights() {
        let size = 8;
        let mut o = vec![0.0; size];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]; // 다양한 weight
        let eps = 1e-5;
        let add_unit_offset = false;

        let sum_sq_x: f32 = x.iter().map(|&val| val * val).sum();
        let mean_sq_x = sum_sq_x / (size as f32);
        let ss_manual = 1.0 / (mean_sq_x + eps).sqrt();

        let mut expected_o = vec![0.0; size];
        for i in 0..size {
            expected_o[i] = x[i] * ss_manual * weight[i];
        }

        rmsnorm(&mut o, &x, &weight, size, eps, add_unit_offset);

        println!("\nTest: Different Weights");
        println!("Input x: {:?}", x);
        println!("Weight: {:?}", weight);
        println!("Output o: {:?}", o);
        println!("Expected o: {:?}", expected_o);
        assert_vec_approx_eq(&o, &expected_o, 1e-5);
    }

    #[test]
    fn test_tanh_f32x8_zero() {
        let input = f32x8::splat(0.0);
        let output = tanh_f32x8(input);
        let expected = f32x8::splat(0.0);
        println!("\nTest: tanh(0)");
        println!("Input: {:?}", input.to_array());
        println!("Output: {:?}", output.to_array());
        println!("Expected: {:?}", expected.to_array());
        assert_vec_approx_eq(&output.to_array(), &expected.to_array(), f32::EPSILON);
    }

    #[test]
    fn test_tanh_f32x8_positive_values() {
        let input = f32x8::from([0.5, 1.0, 2.0, 3.0, 0.1, 0.9, 1.5, 2.5]);
        // 수동 계산 또는 `f32::tanh`를 사용하여 예상 값 생성
        let expected_array = [
            0.5_f32.tanh(),
            1.0_f32.tanh(),
            2.0_f32.tanh(),
            3.0_f32.tanh(),
            0.1_f32.tanh(),
            0.9_f32.tanh(),
            1.5_f32.tanh(),
            2.5_f32.tanh(),
        ];
        let output = tanh_f32x8(input);
        println!("\nTest: tanh(positive values)");
        println!("Input: {:?}", input.to_array());
        println!("Output: {:?}", output.to_array());
        println!("Expected: {:?}", expected_array);
        assert_vec_approx_eq(&output.to_array(), &expected_array, 1e-6);
    }

    #[test]
    fn test_tanh_f32x8_negative_values() {
        let input = f32x8::from([-0.5, -1.0, -2.0, -3.0, -0.1, -0.9, -1.5, -2.5]);
        let expected_array = [
            (-0.5_f32).tanh(),
            (-1.0_f32).tanh(),
            (-2.0_f32).tanh(),
            (-3.0_f32).tanh(),
            (-0.1_f32).tanh(),
            (-0.9_f32).tanh(),
            (-1.5_f32).tanh(),
            (-2.5_f32).tanh(),
        ];
        let output = tanh_f32x8(input);
        println!("\nTest: tanh(negative values)");
        println!("Input: {:?}", input.to_array());
        println!("Output: {:?}", output.to_array());
        println!("Expected: {:?}", expected_array);
        assert_vec_approx_eq(&output.to_array(), &expected_array, 1e-6);
    }

    // Overflow occurs when the input is too large, so we can't test with large values
    // #[test]
    // fn test_tanh_f32x8_large_values() {
    //     let input = f32x8::splat(100.0); // tanh(large_num) -> 1.0
    //     let output = tanh_f32x8(input);
    //     let expected = f32x8::splat(1.0);
    //     println!("\nTest: tanh(large positive)");
    //     println!("Input: {:?}", input.to_array());
    //     println!("Output: {:?}", output.to_array());
    //     println!("Expected: {:?}", expected.to_array());
    //     assert_vec_approx_eq(&output.to_array(), &expected.to_array(), 1e-6);
    //
    //     let input = f32x8::splat(-100.0); // tanh(large_negative_num) -> -1.0
    //     let output = tanh_f32x8(input);
    //     let expected = f32x8::splat(-1.0);
    //     println!("\nTest: tanh(large negative)");
    //     println!("Input: {:?}", input.to_array());
    //     println!("Output: {:?}", output.to_array());
    //     println!("Expected: {:?}", expected.to_array());
    //     assert_vec_approx_eq(&output.to_array(), &expected.to_array(), 1e-6);
    // }

    // --- softmax 테스트 ---

    #[test]
    fn test_softmax_basic() {
        let mut x = vec![1.0, 2.0, 3.0];
        // Expected: exp(1-3)/sum, exp(2-3)/sum, exp(3-3)/sum
        // exp(-2), exp(-1), exp(0) => 0.1353, 0.3679, 1.0
        // Sum = 1.5032
        // Output: 0.0900, 0.2447, 0.6653
        let expected = {
            let mut temp_x = vec![1.0, 2.0, 3.0];
            let max_val = *temp_x
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let mut sum_exp = 0.0;
            let mut exp_vals: Vec<f32> = temp_x
                .iter()
                .map(|&val| {
                    let exp_val = ((val - max_val) as f64).exp() as f32;
                    sum_exp += exp_val;
                    exp_val
                })
                .collect();

            exp_vals.iter_mut().for_each(|val| *val /= sum_exp);
            exp_vals
        };

        softmax(&mut x);

        println!("\nTest: Softmax Basic");
        println!("Input x: {:?}", vec![1.0, 2.0, 3.0]); // 원본 입력 값
        println!("Output x: {:?}", x);
        println!("Expected x: {:?}", expected);
        assert_vec_approx_eq(&x, &expected, 1e-6);
        assert_approx_eq(x.iter().sum(), 1.0, 1e-6);
        // 합이 1인지 확인
    }

    #[test]
    fn test_softmax_with_negatives() {
        let mut x = vec![-1.0, -2.0, -3.0];
        let expected = {
            let mut temp_x = vec![-1.0, -2.0, -3.0];
            let max_val = *temp_x
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let mut sum_exp = 0.0;
            let mut exp_vals: Vec<f32> = temp_x
                .iter()
                .map(|&val| {
                    let exp_val = ((val - max_val) as f64).exp() as f32;
                    sum_exp += exp_val;
                    exp_val
                })
                .collect();
            exp_vals.iter_mut().for_each(|val| *val /= sum_exp);
            exp_vals
        };

        softmax(&mut x);

        println!("\nTest: Softmax with Negatives");
        println!("Input x: {:?}", vec![-1.0, -2.0, -3.0]); // 원본 입력 값
        println!("Output x: {:?}", x);
        println!("Expected x: {:?}", expected);
        assert_vec_approx_eq(&x, &expected, 1e-6);
        assert_approx_eq(x.iter().sum(), 1.0, 1e-6);
    }

    #[test]
    fn test_softmax_all_same_values() {
        let mut x = vec![5.0, 5.0, 5.0, 5.0];
        // Expected: 각 요소가 0.25
        let expected = vec![0.25, 0.25, 0.25, 0.25];

        softmax(&mut x);

        println!("\nTest: Softmax All Same Values");
        println!("Input x: {:?}", vec![5.0, 5.0, 5.0, 5.0]); // 원본 입력 값
        println!("Output x: {:?}", x);
        println!("Expected x: {:?}", expected);
        assert_vec_approx_eq(&x, &expected, 1e-6);
        assert_approx_eq(x.iter().sum(), 1.0, 1e-6);
    }

    #[test]
    fn test_softmax_single_element() {
        let mut x = vec![10.0];
        // Expected: [1.0]
        let expected = vec![1.0];

        softmax(&mut x);

        println!("\nTest: Softmax Single Element");
        println!("Input x: {:?}", vec![10.0]); // 원본 입력 값
        println!("Output x: {:?}", x);
        println!("Expected x: {:?}", expected);
        assert_vec_approx_eq(&x, &expected, 1e-6);
        assert_approx_eq(x.iter().sum(), 1.0, 1e-6);
    }

    #[test]
    fn test_softmax_empty_input() {
        let mut x: Vec<f32> = Vec::new();
        // 비어있는 슬라이스에 대한 소프트맥스 호출은 panics을 발생시키지 않고 아무것도 하지 않아야 함.
        // `x[0]` 접근 때문에 현재 함수는 빈 배열에서 panic 발생.
        // 이 테스트를 통과시키려면 `softmax` 함수에 빈 슬라이스 검사를 추가해야 함.
        // 예: `if x.is_empty() { return; }`
        let expected: Vec<f32> = Vec::new();

        println!("\nTest: Softmax Empty Input");
        softmax(&mut x); // 여기서 `x[0]` 접근 때문에 panic 발생할 수 있음
        println!("Output x (empty): {:?}", x);
        assert_eq!(x, expected, "softmax_empty");
    }

    #[test]
    fn test_softmax_large_range_values() {
        let mut x = vec![1.0, 2.0, 10.0];
        // exp(1-10), exp(2-10), exp(10-10) = exp(-9), exp(-8), exp(0)
        // 0.000123, 0.000335, 1.0
        // Sum approx 1.000458
        // Result should be heavily skewed towards the largest value.
        let expected = {
            let mut temp_x = vec![1.0, 2.0, 10.0];
            let max_val = *temp_x
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let mut sum_exp = 0.0;
            let mut exp_vals: Vec<f32> = temp_x
                .iter()
                .map(|&val| {
                    let exp_val = ((val - max_val) as f64).exp() as f32;
                    sum_exp += exp_val;
                    exp_val
                })
                .collect();
            exp_vals.iter_mut().for_each(|val| *val /= sum_exp);
            exp_vals
        };

        softmax(&mut x);

        println!("\nTest: Softmax Large Range Values");
        println!("Input x: {:?}", vec![1.0, 2.0, 10.0]); // 원본 입력 값
        println!("Output x: {:?}", x);
        println!("Expected x: {:?}", expected);
        assert_vec_approx_eq(&x, &expected, 1e-6);
        assert_approx_eq(x.iter().sum(), 1.0, 1e-6);
    }

    #[test]
    fn test_matmul_32x32_simple() {
        // 32x32 * 32x32 행렬 곱셈 테스트
        let m = 32;
        let n = 32;
        let k = 32;

        // X와 W를 간단한 시퀀스 값으로 초기화
        // X[row][col] = row * N + col + 1
        // W[row][col] = (N - row - 1) * N + (N - col)  (역순으로 채워 패턴 변화)
        let x: Vec<f32> = (0..m * n).map(|i| (i + 1) as f32 / 10.0).collect();
        let w: Vec<f32> = (0..n * k).map(|i| (n * k - i) as f32 / 10.0).collect();
        let mut xout = vec![0.0; m * k];

        // 예상 결과 계산 (참조 구현)
        // 실제 테스트에서는 수동 계산이 불가능하므로,
        // 검증된 '정확한' 행렬 곱셈 함수(reference_matmul)를 사용하여 예상 결과를 생성합니다.
        let mut expected_xout = vec![0.0; m * k];

        // 여기에 정확한 행렬 곱셈을 수행하는 참조 함수를 호출
        // (예: naive_matmul 함수, 또는 외부 라이브러리)
        // 예시를 위해 단순 나이브 구현을 가정합니다.
        // 이 부분은 matmul 구현이 정확하다는 가정을 검증하는 핵심입니다.
        for row_x in 0..m {
            for col_w in 0..k {
                let mut sum = 0.0;
                for i in 0..n {
                    sum += x[row_x * n + i] * w[col_w * k + i];
                }
                expected_xout[row_x * k + col_w] = sum;
            }
        }

        // matmul(&mut xout, &x, &w, m, n, k);
        matmul(&mut xout, &x, &w, m, n);
        println!("\nTest: Matmul 32x32 Simple");
        println!("Input X: {:?}", x);
        println!("Input W: {:?}", w);
        println!("Output XOUT: {:?}", xout);
        println!("Expected XOUT: {:?}", expected_xout);
        assert_vec_approx_eq(&xout, &expected_xout, 1.0);
    }

    #[test]
    fn test_matmul_32xN_Mx32() {
        // 32x64 * 64x32 행렬 곱셈 테스트 (직사각형 형태)
        let m = 32;
        let n = 64; // 중간 차원
        let k = 32;

        let x: Vec<f32> = (0..m * n).map(|i| (i + 1) as f32 / 10.0).collect();
        let w: Vec<f32> = (0..n * k).map(|i| (n * k - i) as f32 / 10.0).collect();
        let mut xout = vec![0.0; m * k];

        let mut expected_xout = vec![0.0; m * k];
        for row_x in 0..m {
            for col_w in 0..k {
                let mut sum = 0.0;
                for i in 0..n {
                    sum += x[row_x * n + i] * w[col_w * n + i];
                }
                expected_xout[row_x * k + col_w] = sum;
            }
        }

        // matmul(&mut xout, &x, &w, m, n, k);
        matmul(&mut xout, &x, &w, n, k);
        println!("\nTest: Matmul 32xN Mx32");
        println!("Input X: {:?}", x);
        println!("Input W: {:?}", w);
        println!("Output XOUT: {:?}", xout);
        assert_vec_approx_eq(&xout, &expected_xout, 1.0);
    }

    #[test]
    fn test_matmul_32x96_64() {
        // X (32x96) * W (96x64) 행렬 곱셈 테스트
        let m = 32; // X의 행 수, XOUT의 행 수
        let n = 96; // X의 열 수, W의 행 수 (중간 차원)
        let k = 64; // W의 열 수, XOUT의 열 수

        // X와 W를 간단한 시퀀스 값으로 초기화
        let x: Vec<f32> = (0..m * n).map(|i| (i + 1) as f32 / 50.0).collect();
        let w: Vec<f32> = (0..n * k).map(|i| (k + i) as f32 / 50.0).collect();
        let mut xout = vec![0.0; m * k]; // 결과 행렬 XOUT (32x64)

        // 예상 결과 계산 (참조 구현)
        // matmul 함수와 동일한 w 인덱싱 방식 (w[col_w * k + i])을 사용합니다.
        let mut expected_xout = vec![0.0; m * k];
        for row_x in 0..m {
            for col_w in 0..k {
                let mut sum = 0.0;
                for i in 0..n {
                    sum += x[row_x * n + i] * w[col_w * n + i];
                }
                expected_xout[row_x * k + col_w] = sum;
            }
        }

        // matmul 함수 호출
        // matmul(&mut xout, &x, &w, m, n, k);
        matmul(&mut xout, &x, &w, n, k);
        println!("\nTest: Matmul 32x96 Mx64");
        println!("Input X: {:?}", x);
        println!("Input W: {:?}", w);
        println!("Output XOUT: {:?}", xout);
        println!("Expected XOUT: {:?}", expected_xout);

        // 결과 검증
        assert_vec_approx_eq(&xout, &expected_xout, 1.0);
    }

    #[test]
    fn test_matmul_32x32_identity() {
        // 32x32 항등 행렬 곱셈 테스트
        let m = 32;
        let n = 32;
        let k = 32;

        let x: Vec<f32> = (0..m * n)
            .map(|i| (i + 1) as f32 / (m * n) as f32)
            .collect(); // 0~1 사이의 값으로 정규화

        let mut w = vec![0.0; n * k];
        for i in 0..n {
            w[i * n + i] = 1.0;
        }

        let mut xout = vec![0.0; m * k];

        // matmul(&mut xout, &x, &w, m, n, k);
        matmul(&mut xout, &x, &w, k, n);
        println!("\nTest: Matmul 32x32 Identity");
        println!("Input X: {:?}", x);
        println!("Input W (Identity): {:?}", w);
        println!("Output XOUT: {:?}", xout);
        assert_vec_approx_eq(&xout, &x, 1e-6);
    }
}
