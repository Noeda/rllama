use crate::tensor::Tensor;
use crate::tokenizer::TokenId;
use rand::Rng;

pub struct TokenSampler {
    temperature: f32,
    top_p: f32,
    top_k: usize,
}

impl Default for TokenSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenSampler {
    pub fn new() -> Self {
        Self {
            temperature: 0.8,
            top_p: 1.0,
            top_k: 1, // same as argmax
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

    pub fn sample(&self, logits: &Tensor) -> TokenId {
        let nrows = logits.rows();
        assert!(logits.cols() == 1);
        let mut logits = logits.transpose();
        if self.temperature > 0.0 {
            logits = logits.scalar_multiply_f32(1.0 / self.temperature);
            logits = logits.softmax();
        }

        let mut logitsf: Vec<(TokenId, f32)> = Vec::with_capacity(nrows as usize);
        for i in 0..nrows {
            logitsf.push((i as TokenId, logits.get_f32(0, i)));
        }
        logitsf.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
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
        let p: f32 = rng.gen_range(0.0..total_p);
        p_accum = 0.0;
        for v in logitsf.into_iter() {
            p_accum += v.1;
            if p_accum >= p {
                return v.0;
            }
        }
        0
    }
}
