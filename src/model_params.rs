use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct ModelParams {
    #[serde(alias = "hidden_size")]
    pub dim: usize,
    #[serde(alias = "num_attention_heads")]
    pub n_heads: usize,
    #[serde(alias = "num_hidden_layers")]
    pub n_layers: usize,
    #[serde(alias = "rms_norm_eps")]
    pub norm_eps: f64,
    pub vocab_size: i64,
}
