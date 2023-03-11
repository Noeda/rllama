use crate::embedding::Embedding;
use crate::token_sampler::TokenSampler;
use crate::tokenizer::{TokenId, Tokenizer};
use crate::transformer::Transformer;
use crate::unpickler;
use clap::Parser;
use colored::Colorize;
use std::io::{Read, Write};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long)]
    model_path: String,
    #[arg(long)]
    tokenizer_path: String,
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long)]
    prompt_file: Option<String>,

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

    let mut be_quiet: bool = false;
    if !colored::control::SHOULD_COLORIZE.should_colorize() {
        be_quiet = true;
    }

    // Custom println-like macro that respects be_quiet
    macro_rules! pln {
        ($($arg:tt)*) => {
            if !be_quiet {
                std::println!($($arg)*);
            }
        };
    }

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
    let mut fs = std::fs::File::open(model_path.as_str())?;
    let mut bs = Vec::new();
    fs.read_to_end(&mut bs)?;
    std::mem::drop(fs);

    // We chop off file name from model_path and append "data/"
    let model_data_dir = model_path
        .split('/')
        .take(model_path.split('/').count() - 1)
        .collect::<Vec<&str>>()
        .join("/")
        + "/data/";
    let result = unpickler::unpickle(&bs)?;
    pln!("Loading embeddings from {}...", model_data_dir);
    let emb = Embedding::from_unpickled(&result, model_data_dir.clone())?;

    let max_seq_len = 512;

    pln!("Loading transformer weights from {}...", model_data_dir);
    let tr = Transformer::from_unpickled(
        &result,
        emb,
        4096,
        32,
        32,
        max_seq_len,
        1e-6,
        32,
        128,
        model_data_dir,
    )?;
    pln!("All is loaded. Starting inference.");

    let mut toks_id: Vec<TokenId> = tok.tokenize_to_ids(prompt.clone());
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

    pln!("---");
    pln!("Temperature: {}", token_sampler.get_temperature());
    pln!("Top P: {}", token_sampler.get_top_p());
    pln!("Top K: {}", token_sampler.get_top_k());
    pln!("---");
    pln!(
        "{}",
        "  This is the color of the initial prompt".truecolor(128, 128, 255)
    );
    pln!(
        "{}",
        "  This is the color of the generated text while full context is available"
            .truecolor(128, 255, 128)
    );
    pln!(
        "{}",
        "  Remaining text is in this color".truecolor(255, 128, 128)
    );
    pln!("---");
    print!("{}", prompt.as_str().truecolor(128, 128, 255));
    let _ = std::io::stdout().flush();

    let mut caches = tr.make_caches();
    let mut first: bool = true;
    let mut shifts: usize = 0;
    loop {
        if toks_id.len() >= max_seq_len {
            toks_id = toks_id[1..].to_vec();
            prev_pos -= 1;
            caches.shift_left(1);
            shifts += 1;
            // TODO: it seems that text beyond context is just broken.
            // Maybe I cannot just go and shift it.
        }
        let preds = tr.forward(&toks_id[prev_pos..], prev_pos, &mut caches, shifts);
        let highest_pred_idx = token_sampler.sample(&preds);
        toks_id.push(highest_pred_idx as TokenId);

        for (tok_idx, tok_id) in toks_id[prev_pos + 1..].iter().enumerate() {
            if *tok_id == 1 {
                continue;
            }
            let mut tok_str: String = "".to_string();
            let tok = tok.id_to_str(*tok_id);
            if tok == "<0x0A>" {
                tok_str += "\n";
            } else {
                tok_str += tok.replace('▁', " ").as_str();
            }
            if first && tok_idx < toks_id.len() - 1 {
                // intentionally left empty
            } else if shifts == 0 {
                print!("{}", tok_str.truecolor(128, 255, 128));
            } else {
                print!("{}", tok_str.truecolor(255, 128, 128));
            }
        }
        let _ = std::io::stdout().flush();
        prev_pos = toks_id.len() - 1;
        first = false;
    }
}
