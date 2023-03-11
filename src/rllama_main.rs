use crate::embedding::Embedding;
use crate::token_sampler::TokenSampler;
use crate::tokenizer::{TokenId, Tokenizer};
use crate::transformer::Transformer;
use crate::unpickler;
use clap::Parser;
use std::io::Read;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long)]
    model_path: String,
    #[arg(long)]
    tokenizer_path: String,
    #[arg(long)]
    prompt: String,

    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    top_p: Option<f32>,
    #[arg(long)]
    top_k: Option<i32>,
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let model_path = cli.model_path;
    let tokenizer_path = cli.tokenizer_path;
    let prompt = cli.prompt;

    println!("Starting up. Loading tokenizer from {}...", tokenizer_path);
    let tok = Tokenizer::load(tokenizer_path.as_str())?;
    println!("Tokenizer loeaded. Loading model from {}...", model_path);
    let mut fs = std::fs::File::open(model_path.as_str())?;
    let mut bs = Vec::new();
    fs.read_to_end(&mut bs)?;
    std::mem::drop(fs);

    // We chop off file name from model_path and append "data/"
    let model_data_dir = model_path
        .split("/")
        .take(model_path.split("/").count() - 1)
        .collect::<Vec<&str>>()
        .join("/")
        + "/data/";
    let result = unpickler::unpickle(&bs)?;
    println!("Loading embeddings from {}...", model_data_dir);
    let emb = Embedding::from_unpickled(&result, model_data_dir.clone())?;

    println!("Loading transformer weights from {}...", model_data_dir);
    let tr = Transformer::from_unpickled(
        &result,
        emb,
        4096,
        32,
        32,
        512,
        1e-6,
        32,
        128,
        model_data_dir,
    )?;
    println!("All is loaded. Starting inference.");

    let mut toks_id: Vec<TokenId> = tok.tokenize_to_ids(prompt);
    let mut prev_pos = 0;
    let mut token_sampler = TokenSampler::new().temperature(0.8).top_p(0.9).top_k(50);

    if let Some(temperature) = cli.temperature {
        token_sampler = token_sampler.temperature(temperature);
    }
    if let Some(top_p) = cli.top_p {
        token_sampler = token_sampler.top_p(top_p);
    }
    if let Some(top_k) = cli.top_k {
        token_sampler = token_sampler.top_k(top_k as usize);
    }

    println!("Temperature: {}", token_sampler.get_temperature());
    println!("Top P: {}", token_sampler.get_top_p());
    println!("Top K: {}", token_sampler.get_top_k());

    let mut caches = tr.make_caches();
    loop {
        let preds = tr.forward(&toks_id[prev_pos..], prev_pos, &mut caches);
        let highest_pred_idx = token_sampler.sample(&preds);
        toks_id.push(highest_pred_idx as TokenId);
        prev_pos = toks_id.len() - 1;

        let mut tok_str: String = "".to_string();
        for tok_id in toks_id.iter() {
            if *tok_id == 1 {
                continue;
            }
            let tok = tok.id_to_str(*tok_id);
            tok_str = tok_str + tok.replace("▁", " ").as_str();
        }
        println!("{}", tok_str);
    }
}
