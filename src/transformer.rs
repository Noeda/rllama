use crate::embedding::Embedding;
use crate::tensor::{Tensor, TensorDType};
use crate::tokenizer::TokenId;
use crate::unpickler;
use crate::unpickler::UnpicklingError;
use indicatif::ProgressBar;
use num_complex::Complex;
use rayon::prelude::*;
use std::path::Path;
use std::sync::{Arc, RwLock};

type FreqsCis = Vec<Vec<Complex<f64>>>;

#[allow(dead_code)]
pub struct Transformer {
    freqs_cis: FreqsCis,
    emb: Embedding,
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_local_heads: usize,
    max_seq_len: usize,
    head_dim: usize,

    norm: RMSNorm,
    output: Tensor,

    layers: Vec<TransformerBlock>,
}

pub struct TransformerCaches {
    layer_caches: Vec<AttentionCache>,
}

pub struct TransformerBlock {
    feed_forward: FeedForward,
    attn: Attention,
    ffn_norm: RMSNorm,
    attention_norm: RMSNorm,
}

pub struct AttentionCache {
    cache_k: Vec<Arc<RwLock<Tensor>>>,
    cache_v: Vec<Arc<RwLock<Tensor>>>,
}

impl AttentionCache {
    fn new(max_seq_len: usize, n_local_heads: usize, head_dim: usize) -> Self {
        let mut cache_k = Vec::with_capacity(n_local_heads);
        let mut cache_v = Vec::with_capacity(n_local_heads);
        for _ in 0..n_local_heads {
            cache_k.push(Arc::new(RwLock::new(Tensor::zeros(
                head_dim as i64,
                max_seq_len as i64,
                TensorDType::Float32,
            ))));
            cache_v.push(Arc::new(RwLock::new(Tensor::zeros(
                head_dim as i64,
                max_seq_len as i64,
                TensorDType::Float32,
            ))));
        }
        AttentionCache { cache_k, cache_v }
    }

    fn shift_left(&mut self, shifts: usize) {
        for _ in 0..shifts {
            for idx in 0..self.cache_k.len() {
                let mut k = self.cache_k[idx].write().unwrap();
                let mut v = self.cache_v[idx].write().unwrap();
                let k_rows = k.rows();
                let k_cols = k.cols();
                for head_idx in 0..k_rows {
                    for seq_idx in 0..k_cols - 1 {
                        let kval = k.get_f32(head_idx, seq_idx + 1);
                        let vval = v.get_f32(head_idx, seq_idx + 1);
                        k.set_f32(head_idx, seq_idx, kval);
                        v.set_f32(head_idx, seq_idx, vval);
                    }
                }
            }
        }
    }
}

impl TransformerCaches {
    pub fn shift_left(&mut self, shifts: usize) {
        for layer in self.layer_caches.iter_mut() {
            layer.shift_left(shifts);
        }
    }
}

pub struct RMSNorm {
    eps: f64,
    weight: Tensor,
}

pub struct Attention {
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    n_local_heads: usize,
    head_dim: usize,
}

pub struct FeedForward {
    w1: Tensor,
    w2: Tensor,
    w3: Tensor,
}

impl Transformer {
    #[allow(clippy::too_many_arguments)]
    pub fn from_unpickled<P: AsRef<Path>>(
        unpickled: &unpickler::Value,
        emb: Embedding,
        dim: usize,
        n_layers: usize,
        n_heads: usize,
        max_seq_len: usize,
        eps: f64,
        n_local_heads: usize,
        head_dim: usize,
        data_dir: P,
    ) -> Result<Transformer, UnpicklingError> {
        let data_dir: &Path = data_dir.as_ref();

        let progress_bar = ProgressBar::new(n_layers as u64);
        let layers: Vec<TransformerBlock> = (0..n_layers)
            .into_par_iter()
            .map(|layer_id| {
                let result = TransformerBlock::from_unpickled(
                    unpickled,
                    layer_id,
                    eps,
                    n_local_heads,
                    head_dim,
                    data_dir,
                );
                progress_bar.inc(1);
                result
            })
            .collect::<Result<Vec<TransformerBlock>, UnpicklingError>>()?;
        std::mem::drop(progress_bar);

        let norm = RMSNorm::from_unpickled(unpickled, "norm.weight".to_string(), eps, data_dir)?;
        let output = Tensor::from_unpickled(unpickled, "output.weight", data_dir)?.to_f32();

        Ok(Transformer {
            // TODO: maybe rotary embedding can be computed on the fly if the sequence gets really long. I just
            // slapped * 20 on max seq len here.
            freqs_cis: compute_freqs_cis(dim / n_heads, max_seq_len * 20, 10000.0),
            emb,
            dim,
            n_layers,
            n_heads,
            n_local_heads,
            max_seq_len,
            head_dim,

            norm,
            output,

            layers,
        })
    }

    pub fn make_caches(&self) -> TransformerCaches {
        let mut result = vec![];
        for _ in 0..self.n_layers {
            result.push(AttentionCache::new(
                self.max_seq_len,
                self.n_local_heads,
                self.head_dim,
            ));
        }
        TransformerCaches {
            layer_caches: result,
        }
    }

    pub fn forward(
        &self,
        tokens: &[TokenId],
        start_pos: usize,
        caches: &mut TransformerCaches,
        shifts: usize,
    ) -> Tensor {
        assert!(caches.layer_caches.len() == self.n_layers);
        let mask: Option<Tensor> = if tokens.len() > 1 {
            Some(Tensor::full_triu(
                tokens.len() as i64,
                tokens.len() as i64,
                start_pos as i64 + 1,
                TensorDType::Float32,
                std::f32::NEG_INFINITY,
            ))
        } else {
            None
        };
        let mut embs: Vec<&Tensor> = Vec::with_capacity(tokens.len());
        for token in tokens.iter() {
            let emb = self.emb.get_embedding(*token as usize);
            embs.push(emb);
        }
        let mut emb_tensor: Tensor = Tensor::concat(&embs);
        std::mem::drop(embs);

        for (idx, layer) in self.layers.iter().enumerate() {
            emb_tensor = layer.forward(
                &emb_tensor,
                start_pos,
                &self.freqs_cis,
                &mask,
                &mut caches.layer_caches[idx],
                shifts,
            );
        }
        let out = self.norm.forward(&emb_tensor);
        let out = out.row(out.rows() - 1);

        self.output.matrix_mul_transposed(&out)
    }
}

impl TransformerBlock {
    pub fn from_unpickled<P: AsRef<Path>>(
        unpickled: &unpickler::Value,
        layer_id: usize,
        eps: f64,
        n_local_heads: usize,
        head_dim: usize,
        data_dir: P,
    ) -> Result<Self, UnpicklingError> {
        let data_dir: &Path = data_dir.as_ref();
        let ff = FeedForward::from_unpickled(unpickled, layer_id, data_dir)?;
        let attn =
            Attention::from_unpickled(unpickled, layer_id, n_local_heads, head_dim, data_dir)?;
        let ffn_norm = RMSNorm::from_unpickled(
            unpickled,
            format!("layers.{}.ffn_norm.weight", layer_id),
            eps,
            data_dir,
        )?;
        let attn_norm = RMSNorm::from_unpickled(
            unpickled,
            format!("layers.{}.attention_norm.weight", layer_id),
            eps,
            data_dir,
        )?;
        Ok(Self {
            feed_forward: ff,
            attn,
            ffn_norm,
            attention_norm: attn_norm,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        freqs_cis: &FreqsCis,
        mask: &Option<Tensor>,
        attention_cache: &mut AttentionCache,
        shifts: usize,
    ) -> Tensor {
        let attnorm_out = self.attention_norm.forward(x);
        let att_out = self.attn.forward(
            &attnorm_out,
            start_pos,
            freqs_cis,
            mask,
            attention_cache,
            shifts,
        );
        let h = x.add(&att_out);
        let att_out = self.ffn_norm.forward(&h);
        let att_out = self.feed_forward.forward(&att_out).transpose();
        h.add(&att_out)
    }
}

impl RMSNorm {
    pub fn from_unpickled<P: AsRef<Path>>(
        unpickled: &unpickler::Value,
        name: String,
        eps: f64,
        data_dir: P,
    ) -> Result<RMSNorm, UnpicklingError> {
        let data_dir: &Path = data_dir.as_ref();
        let weights = Tensor::from_unpickled(unpickled, name, data_dir)?.to_f32();
        Ok(Self {
            eps,
            weight: weights,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let inner = x.pow(2.0).mean_cols().add_scalar(self.eps as f32);
        let out1 = x.scalar_multiply_broadcast(&inner.rsqrt());
        out1.hadamard_product_broadcast(&self.weight)
    }
}

impl FeedForward {
    pub fn from_unpickled<P: AsRef<Path>>(
        unpickled: &unpickler::Value,
        layer_id: usize,
        data_dir: P,
    ) -> Result<FeedForward, UnpicklingError> {
        let data_dir: &Path = data_dir.as_ref();

        let w1 = Tensor::from_unpickled(
            unpickled,
            format!("layers.{}.feed_forward.w1.weight", layer_id),
            data_dir,
        )?
        .to_f32();
        let w2 = Tensor::from_unpickled(
            unpickled,
            format!("layers.{}.feed_forward.w2.weight", layer_id),
            data_dir,
        )?
        .to_f32();
        let w3 = Tensor::from_unpickled(
            unpickled,
            format!("layers.{}.feed_forward.w3.weight", layer_id),
            data_dir,
        )?
        .to_f32();

        Ok(Self { w1, w2, w3 })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (w1_out, w3_out) = rayon::join(
            || self.w1.matrix_mul_transposed(x),
            || self.w3.matrix_mul_transposed(x),
        );
        let w1_out = w1_out.silu();
        let w1w3_out = w1_out.hadamard_product(&w3_out).transpose();

        if w1w3_out.rows() == 1 {
            return self
                .w2
                .matrix_vector_mul_transposed_multithreaded(&w1w3_out);
        }
        self.w2.matrix_mul_transposed(&w1w3_out)
    }
}

impl Attention {
    pub fn from_unpickled<P: AsRef<Path>>(
        unpickled: &unpickler::Value,
        layer_id: usize,
        n_local_heads: usize,
        head_dim: usize,
        data_dir: P,
    ) -> Result<Attention, UnpicklingError> {
        let data_dir: &Path = data_dir.as_ref();

        let wq = Tensor::from_unpickled(
            unpickled,
            format!("layers.{}.attention.wq.weight", layer_id),
            data_dir,
        )?
        .to_f32();
        let wk = Tensor::from_unpickled(
            unpickled,
            format!("layers.{}.attention.wk.weight", layer_id),
            data_dir,
        )?
        .to_f32();
        let wv = Tensor::from_unpickled(
            unpickled,
            format!("layers.{}.attention.wv.weight", layer_id),
            data_dir,
        )?
        .to_f32();
        let wo = Tensor::from_unpickled(
            unpickled,
            format!("layers.{}.attention.wo.weight", layer_id),
            data_dir,
        )?
        .to_f32();

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            n_local_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        freqs_cis: &FreqsCis,
        mask: &Option<Tensor>,
        attention_cache: &mut AttentionCache,
        shifts: usize,
    ) -> Tensor {
        let seq_len = x.rows();
        let (xq_out, (xk_out, xv_out)) = rayon::join(
            || x.matrix_mul_transposed(&self.wq),
            || {
                rayon::join(
                    || x.matrix_mul_transposed(&self.wk),
                    || x.matrix_mul_transposed(&self.wv),
                )
            },
        );

        let mut xq_views: Vec<Tensor> = Vec::with_capacity(seq_len as usize);
        let mut xk_views: Vec<Tensor> = Vec::with_capacity(seq_len as usize);
        let mut xv_views: Vec<Tensor> = Vec::with_capacity(seq_len as usize);

        for idx in 0..seq_len {
            let xq_row = xq_out
                .row(idx)
                .view(self.n_local_heads as i64, self.head_dim as i64);
            let xk_row = xk_out
                .row(idx)
                .view(self.n_local_heads as i64, self.head_dim as i64);
            let xv_row = xv_out
                .row(idx)
                .view(self.n_local_heads as i64, self.head_dim as i64);

            let (xq_row, xk_row) = apply_rotary_emb(
                &xq_row,
                &xk_row,
                freqs_cis,
                idx as usize,
                start_pos + shifts,
            );

            xq_views.push(xq_row);
            xk_views.push(xk_row);
            xv_views.push(xv_row);
        }

        let output: Vec<Tensor> = (0..self.n_local_heads)
            .into_par_iter()
            .map(|idx| {
                let mut concat_vec: Vec<Tensor> = vec![];
                for idx2 in 0..seq_len {
                    concat_vec.push(xq_views[idx2 as usize].row(idx as i64));
                }
                let concat_vec2: Vec<&Tensor> = concat_vec.iter().collect();
                let xq_row = Tensor::concat(&concat_vec2);

                concat_vec.truncate(0);
                for idx2 in 0..seq_len {
                    concat_vec.push(xk_views[idx2 as usize].row(idx as i64));
                }
                let concat_vec2: Vec<&Tensor> = concat_vec.iter().collect();
                let xk_row = Tensor::concat(&concat_vec2).transpose();

                concat_vec.truncate(0);
                for idx2 in 0..seq_len {
                    concat_vec.push(xv_views[idx2 as usize].row(idx as i64));
                }
                let concat_vec2: Vec<&Tensor> = concat_vec.iter().collect();
                let xv_row = Tensor::concat(&concat_vec2);

                let mut cache_k = attention_cache.cache_k[idx].write().unwrap();
                let mut cache_v = attention_cache.cache_v[idx].write().unwrap();

                for pos in start_pos..start_pos + seq_len as usize {
                    for dim in 0..self.head_dim {
                        let k = xk_row.get_f32(dim as i64, (pos - start_pos) as i64);
                        cache_k.set_f32(dim as i64, pos as i64, k);
                        let v = xv_row.get_f32((pos - start_pos) as i64, dim as i64);
                        cache_v.set_f32(dim as i64, pos as i64, v);
                    }
                }
                let keys = cache_k.clip_cols(start_pos + seq_len as usize);
                let values = cache_v.clip_cols(start_pos + seq_len as usize);

                let m = xq_row
                    .matrix_mul(&keys)
                    .scalar_multiply_f32(1.0 / (self.head_dim as f32).sqrt());

                match mask {
                    Some(ref mask) => m
                        .add(mask)
                        .to_f32()
                        .softmax()
                        .matrix_mul_transposed(&values),
                    None => m.softmax().matrix_mul_transposed(&values),
                }
            })
            .collect();

        let output2: Vec<Tensor> = (0..seq_len)
            .into_par_iter()
            .map(|idx| {
                let mut concat_vec: Vec<Tensor> = vec![];
                for output in &output {
                    concat_vec.push(output.row(idx));
                }
                let concat_vec2: Vec<&Tensor> = concat_vec.iter().collect();
                let xq_row = Tensor::concat(&concat_vec2).view(1, 4096);
                xq_row.matrix_mul_transposed(&self.wo)
            })
            .collect();

        let output3: Vec<&Tensor> = output2.iter().collect();
        let output2: Tensor = Tensor::concat(&output3);
        output2
    }
}

fn apply_rotary_emb(
    xq: &Tensor,
    xk: &Tensor,
    freqs_cis: &FreqsCis,
    seq_idx: usize,
    start_pos: usize,
) -> (Tensor, Tensor) {
    assert!(xq.cols() % 2 == 0);
    assert!(xk.cols() % 2 == 0);
    let mut xq_out: Tensor = xq.clone();
    let mut xk_out: Tensor = xk.clone();
    for row in 0..xq.rows() {
        for col in 0..xq.cols() / 2 {
            let f_real = freqs_cis[seq_idx + start_pos][col as usize].re as f32;
            let f_imag = freqs_cis[seq_idx + start_pos][col as usize].im as f32;
            let xq_real = xq.get_f32(row, col * 2);
            let xq_imag = xq.get_f32(row, col * 2 + 1);
            let xk_real = xk.get_f32(row, col * 2);
            let xk_imag = xk.get_f32(row, col * 2 + 1);

            // multiply with freqs_cis
            let xq_realpart = xq_real * f_real - xq_imag * f_imag;
            let xq_imagpart = xq_real * f_imag + xq_imag * f_real;
            let xk_realpart = xk_real * f_real - xk_imag * f_imag;
            let xk_imagpart = xk_real * f_imag + xk_imag * f_real;

            xq_out.set_f32(row, col * 2, xq_realpart);
            xq_out.set_f32(row, col * 2 + 1, xq_imagpart);
            xk_out.set_f32(row, col * 2, xk_realpart);
            xk_out.set_f32(row, col * 2 + 1, xk_imagpart);
        }
    }
    (xq_out, xk_out)
}

fn compute_freqs_cis(dim: usize, end: usize, theta: f64) -> FreqsCis {
    let mut freqs = Vec::new();
    for idx in 0..(dim / 2) {
        let freq = 1.0 / (theta.powf(idx as f64 * 2.0 / dim as f64));
        freqs.push(freq);
    }

    let mut result: Vec<Vec<f64>> = Vec::new();
    for x in 0..end {
        let mut row = Vec::new();
        for freq in freqs.iter() {
            let freq = freq * (x as f64);
            row.push(freq);
        }
        result.push(row);
    }

    let mut resultc: Vec<Vec<Complex<f64>>> = Vec::new();
    for row in result.into_iter() {
        let mut rowc = Vec::new();
        for freq in row {
            let cis = Complex::from_polar(1.0, freq);
            rowc.push(cis);
        }
        resultc.push(rowc);
    }
    resultc
}
