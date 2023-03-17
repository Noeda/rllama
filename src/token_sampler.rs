use crate::tensor::Tensor;
use crate::tokenizer::{TokenId, Tokenizer};
use rand::Rng;
use std::collections::BTreeMap;

pub struct TokenSampler {
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repetition_penalty: f32,
}

impl Default for TokenSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenSampler {
    pub fn new() -> Self {
        Self {
            temperature: 0.2,
            top_p: 1.0,
            top_k: 1, // same as argmax
            repetition_penalty: 0.8, // 1.0 = no penalty. values above 1.0 make repetition
                      // encouraged which can quickly devolve into repeating loop
        }
    }

    pub fn get_temperature(&self) -> f32 {
        self.temperature
    }

    pub fn get_top_p(&self) -> f32 {
        self.top_p
    }

    pub fn get_top_k(&self) -> usize {
        self.top_k
    }

    pub fn get_repetition_penalty(&self) -> f32 {
        self.repetition_penalty
    }

    pub fn temperature(self, temperature: f32) -> Self {
        Self {
            temperature,
            ..self
        }
    }

    pub fn top_p(self, top_p: f32) -> Self {
        Self { top_p, ..self }
    }

    pub fn top_k(self, top_k: usize) -> Self {
        Self { top_k, ..self }
    }

    pub fn repetition_penalty(self, repetition_penalty: f32) -> Self {
        Self {
            repetition_penalty,
            ..self
        }
    }

    pub fn sample(
        &self,
        logits: &Tensor,
        _tokenizer: &Tokenizer,
        existing_tokens: &[TokenId],
    ) -> (TokenId, f32) {
        let mut times_used: BTreeMap<TokenId, usize> = BTreeMap::new();
        for token in existing_tokens {
            times_used
                .entry(*token)
                .and_modify(|e| *e += 1)
                .or_insert(1);
        }

        let nrows = logits.rows();
        assert!(logits.cols() == 1);
        let mut logits = logits.transpose();
        if self.temperature > 0.0 {
            logits = logits.scalar_multiply_f32(1.0 / self.temperature);
        }

        if self.repetition_penalty != 1.0 {
            for token_idx in 0..logits.rows() {
                if let Some(count) = times_used.get(&(token_idx as TokenId)) {
                    let penalty = self.repetition_penalty.powf(*count as f32);
                    logits.set_f32(0, token_idx, logits.get_f32(0, token_idx) * penalty);
                }
            }
        }
        let mut maxv: f32 = std::f32::NEG_INFINITY;
        for token_idx in 0..logits.rows() {
            let v = logits.get_f32(0, token_idx);
            if v > maxv {
                maxv = v;
            }
        }
        // To numerically stabilize, remove maxv from all logits
        // softmax(x + c) = softmax(x) where c is a constant, and we make use of htat
        for token_idx in 0..logits.rows() {
            logits.set_f32(0, token_idx, logits.get_f32(0, token_idx) - maxv);
        }
        logits = logits.softmax();

        let mut logitsf: Vec<(TokenId, f32)> = Vec::with_capacity(nrows as usize);
        for i in 0..nrows {
            let score = logits.get_f32(0, i);
            logitsf.push((i as TokenId, score));
        }
        logitsf.sort_unstable_by(|a, b| {
            match b.1.partial_cmp(&a.1) {
                Some(c) => c,
                None => {
                    // Sort NaNs to bottom
                    if b.1.is_nan() {
                        std::cmp::Ordering::Less
                    } else if a.1.is_nan() {
                        return std::cmp::Ordering::Greater;
                    } else {
                        return std::cmp::Ordering::Equal;
                    }
                }
            }
        });

        logitsf.truncate(self.top_k);
        let mut p_accum: f32 = 0.0;
        for (idx, v) in logitsf.iter().enumerate() {
            p_accum += v.1;
            if p_accum >= self.top_p {
                logitsf.truncate(idx + 1);
                break;
            }
        }
        let mut total_p: f32 = 0.0;
        for v in logitsf.iter() {
            total_p += v.1;
        }
        let mut rng = rand::thread_rng();
        let p: f32 = rng.gen_range(0.0..=total_p);
        p_accum = 0.0;
        for v in logitsf.into_iter() {
            p_accum += v.1;
            if p_accum >= p {
                return (v.0, v.1 / total_p);
            }
        }
        (0, 0.0)
    }
}
