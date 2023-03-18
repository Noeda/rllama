use crate::embedding::Embedding;
#[cfg(feature = "opencl")]
use crate::tensor_opencl_support::OpenCL;
use crate::token_sampler::TokenSampler;
use crate::tokenizer::{TokenId, Tokenizer};
use crate::transformer::{DataSettings, Transformer};
use crate::unpickler;
use crate::unpickler::Value;
use clap::Parser;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long)]
    model_path: String,
    #[arg(long)]
    tokenizer_path: String,
    #[arg(long)]
    param_path: String,

    #[arg(long)]
    prompt: Option<String>,
    #[arg(long)]
    prompt_file: Option<String>,

    #[arg(long)]
    max_seq_len: Option<usize>,

    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    top_p: Option<f32>,
    #[arg(long)]
    top_k: Option<i32>,
    #[arg(long)]
    repetition_penalty: Option<f32>,

    #[arg(long)]
    max_threads: Option<usize>,

    #[arg(long, action)]
    f16: bool,

    #[cfg(feature = "opencl")]
    #[arg(long)]
    opencl_device: Option<usize>,
}

#[derive(Clone, Serialize, Deserialize)]
struct ModelParams {
    dim: usize,
    multiple_of: usize,
    n_heads: usize,
    n_layers: usize,
    norm_eps: f64,
    vocab_size: i64,
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let model_path = cli.model_path;
    let tokenizer_path = cli.tokenizer_path;
    let param_path = cli.param_path;

    let max_threads: usize = match cli.max_threads {
        None => rayon::current_num_threads(),
        Some(max_threads) => {
            rayon::ThreadPoolBuilder::new()
                .num_threads(max_threads)
                .build_global()
                .unwrap();
            max_threads
        }
    };

    let mut be_quiet: bool = false;
    if !colored::control::SHOULD_COLORIZE.should_colorize() {
        be_quiet = true;
    }

    #[cfg(feature = "opencl")]
    let opencl: Option<OpenCL> = {
        let opencl_device = cli.opencl_device.unwrap_or(0);
        match OpenCL::new(!be_quiet, opencl_device) {
            Err(openclerr) => {
                eprintln!("OpenCL error: {}", openclerr);
                eprintln!("OpenCL is disabled because it failed to initialize.");
                None
            }
            Ok(opencl) => {
                println!("OpenCL initialized.");
                Some(opencl)
            }
        }
    };

    // Custom println-like macro that respects be_quiet
    macro_rules! pln {
        ($($arg:tt)*) => {
            if !be_quiet {
                std::println!($($arg)*);
            }
        };
    }

    // Read ModelParams from param_path, we expect it to be JSON
    let mut fs = std::fs::File::open(&param_path)?;
    let mut bs = Vec::new();
    fs.read_to_end(&mut bs)?;
    std::mem::drop(fs);
    let params: ModelParams = serde_json::from_slice(&bs)?;
    pln!("Loaded model parameters from {}.", param_path);

    let prompt: String = match (cli.prompt, cli.prompt_file) {
        (Some(prompt), None) => {
            pln!("Using prompt: {}", prompt);
            prompt
        }
        (None, Some(prompt_file)) => {
            pln!("Using prompt file: {}", prompt_file);
            let mut fs = std::fs::File::open(prompt_file)?;
            let mut bs = Vec::new();
            fs.read_to_end(&mut bs)?;
            std::mem::drop(fs);
            String::from_utf8(bs)?
        }
        _ => {
            eprintln!("Please provide either a prompt or a prompt file.");
            return Err("Please provide either a prompt or a prompt file.".into());
        }
    };

    pln!("Starting up. Loading tokenizer from {}...", tokenizer_path);
    let tok = Tokenizer::load(tokenizer_path.as_str())?;
    pln!("Tokenizer loaded. Loading model from {}...", model_path);

    let mut unpickle_results: Vec<Value> = vec![];

    let mut part: usize = 0;
    loop {
        let model_path: PathBuf = model_path.clone().into();
        let base_path = model_path.join(format!("consolidated.{:02}", part));
        // The data file is in consolidated.XX/data.pkl where XX is the part number.
        let full_path = base_path.join("data.pkl");
        let mut fs = match std::fs::File::open(&full_path) {
            Ok(fs) => fs,
            Err(err) => {
                if err.kind() == std::io::ErrorKind::NotFound {
                    break;
                } else {
                    return Err(err.into());
                }
            }
        };
        let mut bs = Vec::new();
        fs.read_to_end(&mut bs)?;
        std::mem::drop(fs);
        pln!("Read data.pkl from path {}", full_path.display());

        let result = unpickler::unpickle(&bs)?;
        unpickle_results.push(result);
        part += 1;
    }

    pln!("Loading embeddings from {}...", model_path);
    let emb = Embedding::from_unpickled(&unpickle_results, model_path.clone())?;

    let max_seq_len = cli.max_seq_len.unwrap_or(1024);

    let mut data_settings = {
        #[cfg(feature = "opencl")]
        {
            if let Some(opencl) = opencl {
                let ds = DataSettings::new(Some(opencl));
                ds.use_opencl()
            } else {
                DataSettings::new(None)
            }
        }
        #[cfg(not(feature = "opencl"))]
        DataSettings::new()
    };

    if cli.f16 {
        data_settings = data_settings.force_f16();
    }

    pln!("Loading transformer weights from {}...", model_path);
    let tr = Transformer::from_unpickled(
        &unpickle_results,
        emb,
        params.dim,
        params.n_layers,
        params.n_heads,
        max_seq_len,
        params.norm_eps,
        data_settings,
        model_path,
    )?;
    pln!("All is loaded. Starting inference.");

    let mut toks_id: Vec<TokenId> = tok.tokenize_to_ids(prompt.clone());
    let mut prev_pos = 0;
    let mut token_sampler = TokenSampler::new()
        .temperature(0.8)
        .top_p(0.9)
        .top_k(50)
        .repetition_penalty(0.8);

    if let Some(temperature) = cli.temperature {
        token_sampler = token_sampler.temperature(temperature);
    }
    if let Some(top_p) = cli.top_p {
        token_sampler = token_sampler.top_p(top_p);
    }
    if let Some(top_k) = cli.top_k {
        token_sampler = token_sampler.top_k(top_k as usize);
    }
    if let Some(repetition_penalty) = cli.repetition_penalty {
        token_sampler = token_sampler.repetition_penalty(repetition_penalty);
    }

    pln!("---");
    pln!(" dim: {}", params.dim);
    pln!(" multiple_of: {}", params.multiple_of);
    pln!(" n_heads: {}", params.n_heads);
    pln!(" n_layers: {}", params.n_layers);
    pln!(" norm_eps: {}", params.norm_eps);
    pln!(" vocab_size: {}", params.vocab_size);
    pln!("---");
    pln!(" maximum number of threads: {}", max_threads);
    pln!("---");
    pln!("Max sequence length: {}", max_seq_len);
    pln!("Temperature: {}", token_sampler.get_temperature());
    pln!("Top P: {}", token_sampler.get_top_p());
    pln!("Top K: {}", token_sampler.get_top_k());
    pln!(
        "Repetition penalty: {}",
        token_sampler.get_repetition_penalty()
    );
    pln!("---");
    pln!(
        "{}",
        "  This is the color of the initial prompt".truecolor(128, 128, 255)
    );
    pln!(
        "{}",
        "  This is the color of the generated text".truecolor(128, 255, 128)
    );
    pln!("---");
    print!("{}", prompt.as_str().truecolor(128, 128, 255));
    let _ = std::io::stdout().flush();

    let mut first_token_time: std::time::Duration = std::time::Duration::new(0, 0);
    let mut times_per_token: Vec<std::time::Duration> = vec![];
    let mut caches = tr.make_caches();
    let mut first: bool = true;
    let mut stop_seen: bool = false;
    while toks_id.len() < max_seq_len {
        let now = std::time::Instant::now();
        let preds = tr.forward(&toks_id[prev_pos..], prev_pos, &mut caches);

        let (highest_pred_idx, token_prob) = token_sampler.sample(&preds, &tok, &toks_id);
        toks_id.push(highest_pred_idx as TokenId);

        for (tok_idx, tok_id) in toks_id[prev_pos + 1..].iter().enumerate() {
            if *tok_id == 1 {
                continue;
            }
            let mut tok_str: String = "".to_string();
            let tok = tok.id_to_str(*tok_id);
            if tok == "</s>" {
                tok_str += "";
                stop_seen = true;
            }
            if tok == "<0x0A>" {
                tok_str += "\n";
            } else {
                tok_str += tok.replace('▁', " ").as_str();
            }
            if first && tok_idx < toks_id.len() - 2 {
                // intentionally left empty
            } else {
                let redness: f32 = token_prob * 255.0;
                let redness = if redness > 255.0 {
                    255
                } else if redness < 0.0 {
                    0
                } else {
                    redness as u8
                };
                print!(
                    "{}",
                    tok_str.truecolor(128 + redness / 2, 255 - redness / 2, 128)
                );
            }
        }
        if first {
            first_token_time = now.elapsed();
        } else {
            times_per_token.push(now.elapsed());
        }
        let _ = std::io::stdout().flush();
        prev_pos = toks_id.len() - 1;
        first = false;
        if stop_seen {
            break;
        }
    }
    println!();
    if stop_seen && !be_quiet {
        println!("Stop token seen. Stopping.");
    }
    if !be_quiet {
        println!("---");
        println!(
            "Time taken to generate first token: {:?}ms",
            first_token_time.as_millis()
        );
        println!(
            "Time taken per token (excluding first token): {:?}ms",
            times_per_token.iter().map(|t| t.as_millis()).sum::<u128>()
                / times_per_token.len() as u128
        );
    }
    Ok(())
}
