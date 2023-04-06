use crate::data_source::DataSource;
use crate::embedding::Embedding;
use crate::model_params::ModelParams;

#[cfg(feature = "opencl")]
use crate::tensor_opencl_support::OpenCL;
use crate::token_sampler::TokenSampler;
use crate::tokenizer::{TokenId, Tokenizer};
use crate::transformer::{DataSettings, Transformer};

#[cfg(feature = "server")]
use crate::semaphore::Semaphore;
#[cfg(feature = "server")]
use crate::transformer::TransformerCaches;
use clap::Parser;
use colored::Colorize;
#[cfg(feature = "server")]
use rocket::{response::status, response::Stream, Data, State};
use serde::{Deserialize, Serialize};
#[cfg(feature = "server")]
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::sync::Arc;
#[cfg(feature = "server")]
use std::sync::RwLock;

// Refer to README.md to see what all these options mean.
#[derive(Parser, Clone)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long)]
    model_path: String,
    #[arg(long)]
    tokenizer_path: String,
    #[arg(long)]
    param_path: String,

    #[arg(short, long, action)]
    quiet: bool,

    #[arg(long)]
    prompt: Option<String>,
    #[arg(long)]
    prompt_file: Option<String>,

    #[arg(long)]
    interactive_system_prompt: Option<String>,
    #[arg(long)]
    interactive_stop: Vec<String>,
    #[arg(long)]
    interactive_prompt_postfix: Option<String>,
    #[arg(long)]
    interactive_prompt_prefix: Option<String>,
    #[arg(long, action)]
    start_interactive: bool,

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

    #[cfg(feature = "opencl")]
    #[arg(long)]
    percentage_to_gpu: Option<f32>,

    #[arg(long, action)]
    inference_server: bool,

    #[arg(long)]
    inference_server_port: Option<u16>,

    #[arg(long)]
    inference_server_host: Option<String>,

    #[arg(long)]
    inference_server_max_concurrent_inferences: Option<usize>,

    #[arg(long)]
    inference_server_api_path: Option<String>,

    #[arg(long)]
    inference_server_prompt_cache_size: Option<usize>,

    #[arg(long, action)]
    inference_server_exit_after_one_query: bool,
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let model_path = cli.model_path.clone();
    let tokenizer_path = cli.tokenizer_path.clone();
    let param_path = cli.param_path.clone();
    let interactive_system_prompt = cli.interactive_system_prompt.clone().unwrap_or("A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, terse answers to the human's questions.### Human:".to_string());
    let mut interactive_stop = cli.interactive_stop.clone();
    if interactive_stop.is_empty() {
        // Desperado to catch all weird variants of ###Human the model might spit out.
        interactive_stop = vec![
            "### Human:".to_string(),
            "###Human:".to_string(),
            "### Human: ".to_string(),
            "###Human: ".to_string(),
            " ### Human:".to_string(),
            " ###Human:".to_string(),
            " ### Human: ".to_string(),
            " ###Human: ".to_string(),
            "\n### Human:".to_string(),
            "\n###Human:".to_string(),
            "\n### Human: ".to_string(),
            "\n###Human: ".to_string(),
            "\n ### Human:".to_string(),
            "\n ###Human:".to_string(),
            "\n ### Human: ".to_string(),
            "\n ###Human: ".to_string(),
        ];
    }
    let interactive_prompt_prefix = cli
        .interactive_prompt_prefix
        .clone()
        .unwrap_or(" ".to_string());
    let interactive_prompt_postfix = cli
        .interactive_prompt_postfix
        .clone()
        .unwrap_or("### Assistant:".to_string());
    let start_interactive = cli.start_interactive;
    #[cfg(not(feature = "server"))]
    if cli.inference_server {
        eprintln!("Inference server is not enabled in this build.");
        return Err("Inference server is not enabled in this build.".into());
    }

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

    #[cfg(feature = "opencl")]
    let percentage_to_gpu: f32 = cli.percentage_to_gpu.unwrap_or(1.0);

    let mut be_quiet: bool = false;
    if !colored::control::SHOULD_COLORIZE.should_colorize() {
        be_quiet = true;
    }
    if cli.quiet {
        be_quiet = true;
    }
    if be_quiet {
        colored::control::SHOULD_COLORIZE.set_override(false);
    }

    // Custom println-like macro that respects be_quiet
    macro_rules! pln {
        ($($arg:tt)*) => {
            if !be_quiet {
                std::println!($($arg)*);
            }
        };
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

    #[cfg(feature = "opencl")]
    let has_opencl = opencl.is_some();

    // Read ModelParams from param_path, we expect it to be JSON
    let mut fs = std::fs::File::open(&param_path)?;
    let mut bs = Vec::new();
    fs.read_to_end(&mut bs)?;
    std::mem::drop(fs);

    let prompt: String = match (&cli.prompt, &cli.prompt_file, start_interactive) {
        (Some(ref prompt), None, _) => {
            pln!("Using prompt: {}", prompt);
            prompt.clone()
        }
        (None, Some(ref prompt_file), _) => {
            pln!("Using prompt file: {}", prompt_file);
            let mut fs = std::fs::File::open(prompt_file)?;
            let mut bs = Vec::new();
            fs.read_to_end(&mut bs)?;
            std::mem::drop(fs);
            String::from_utf8(bs)?
        }
        (_, _, false) => {
            if cli.inference_server {
                "".to_string()
            } else {
                eprintln!("Please provide either a prompt or a prompt file.");
                return Err("Please provide either a prompt or a prompt file.".into());
            }
        }
        (None, None, true) => "".to_string(),
        (_, _, true) => {
            eprintln!("Please provide either a prompt or a prompt file.");
            return Err("Please provide either a prompt or a prompt file.".into());
        }
    };

    pln!("Starting up. Loading tokenizer from {}...", tokenizer_path);
    let tok = Tokenizer::load(tokenizer_path.as_str())?;
    pln!("Tokenizer loaded. Loading model from {}...", model_path);

    let model_data_source = DataSource::from_inferred_source(model_path.clone())?;

    let params: ModelParams = serde_json::from_slice(&bs)?;
    pln!("Loaded model parameters from {}.", param_path);

    pln!("Loading embeddings from {}...", model_path);
    let emb = Embedding::from_unpickled(model_data_source.clone())?;

    let max_seq_len = cli.max_seq_len.unwrap_or(1024);

    let mut data_settings = {
        #[cfg(feature = "opencl")]
        {
            if let Some(opencl) = opencl {
                let ds = DataSettings::new(Some(opencl));
                ds.percentage_to_gpu(percentage_to_gpu).use_opencl()
            } else {
                DataSettings::new(None)
            }
        }
        #[cfg(not(feature = "opencl"))]
        DataSettings::new()
    };

    #[cfg(feature = "opencl")]
    if cli.f16 || has_opencl {
        data_settings = data_settings.force_f16();
    }
    #[cfg(not(feature = "opencl"))]
    if cli.f16 {
        data_settings = data_settings.force_f16();
    }

    pln!("Loading transformer weights from {}...", model_path);
    let tr = Transformer::from_unpickled(
        emb,
        params.dim,
        params.n_layers,
        params.n_heads,
        max_seq_len,
        params.norm_eps,
        data_settings,
        model_data_source,
    )?;
    pln!("All is loaded. Starting inference.");

    let tr: Arc<Transformer> = Arc::new(tr);
    let tok: Arc<Tokenizer> = Arc::new(tok);

    if cli.inference_server {
        #[cfg(feature = "server")]
        {
            server_inference(cli, tr, tok, be_quiet, max_seq_len, params, max_threads)
        }
        #[cfg(not(feature = "server"))]
        {
            eprintln!("The inference server feature is not enabled.");
            eprintln!("Please enable it with the \"inference-server\" feature.");
            Err("The inference server feature is not enabled.".into())
        }
    } else {
        command_line_inference(
            cli.clone(),
            tr.clone(),
            tok.clone(),
            prompt.clone(),
            interactive_stop.clone(),
            interactive_system_prompt.clone(),
            interactive_prompt_prefix.clone(),
            interactive_prompt_postfix.clone(),
            start_interactive,
            be_quiet,
            max_seq_len,
            params.clone(),
            max_threads,
        )
    }
}

#[cfg(feature = "server")]
fn server_inference(
    cli: Cli,
    tr: Arc<Transformer>,
    tok: Arc<Tokenizer>,
    be_quiet: bool,
    max_seq_len: usize,
    _params: ModelParams,
    _max_threads: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    macro_rules! pln {
        ($($arg:tt)*) => {
            if !be_quiet {
                std::println!($($arg)*);
            }
        };
    }

    let inference_server_port = cli.inference_server_port.unwrap_or(8080);
    let inference_server_host = cli
        .inference_server_host
        .clone()
        .unwrap_or("127.0.0.1".to_string());
    let inference_server_max_concurrent_inferences =
        cli.inference_server_max_concurrent_inferences.unwrap_or(5);
    let inference_server_api_path = cli
        .inference_server_api_path
        .clone()
        .unwrap_or("/rllama/v1/inference".to_string());
    let inference_server_prompt_cache_size = cli.inference_server_prompt_cache_size.unwrap_or(50);

    pln!(
        "Maximum concurrent inferences: {}",
        inference_server_max_concurrent_inferences
    );
    pln!("Prompt cache size: {}", inference_server_prompt_cache_size);
    pln!("Maximum sequence length: {}", max_seq_len);
    pln!(
        "--- Starting HTTP server on {}:{}, answering to requests at {} ---",
        inference_server_host,
        inference_server_port,
        inference_server_api_path
    );

    // If there are too many connections, they will hang until they get their turn.
    // Maybe can later implement return 503 slow down or something similar.
    let concurrent_requests_semaphore = Semaphore::new(inference_server_max_concurrent_inferences);

    let rocket_conf = rocket::Config::build(rocket::config::Environment::Production)
        .address(inference_server_host)
        .port(inference_server_port)
        .finalize()
        .unwrap();

    let app = rocket::custom(rocket_conf)
        .mount(&inference_server_api_path, routes![handle_request])
        .manage(InferenceServerState {
            transformer: tr,
            tokenizer: tok,
            max_seq_len,
            concurrent_requests_semaphore,
            attention_cache_repository: Arc::new(RwLock::new(AttentionCacheRepository::empty(
                inference_server_prompt_cache_size,
            ))),
            exit_after_one_query: cli.inference_server_exit_after_one_query,
        });

    app.launch();
    panic!("Starting web server failed.");
}

#[cfg(feature = "server")]
fn is_false(b: &bool) -> bool {
    !b
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct InferenceRequest {
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    max_seq_len: Option<usize>,
    max_new_tokens: Option<usize>,
    no_token_sampling: Option<bool>,
    stop_at_end_token: Option<bool>,
    prompt: String,
}

#[cfg(feature = "server")]
#[derive(Serialize, Deserialize, Clone, Debug)]
struct PredResult {
    p: f32,
    #[serde(skip_serializing_if = "is_false")]
    is_end_token: bool,
}

#[cfg(feature = "server")]
struct GeneratingSession {
    transformer: Arc<Transformer>,
    token_sampler: TokenSampler,
    tokenizer: Arc<Tokenizer>,
    attention_cache_repository: Arc<RwLock<AttentionCacheRepository>>,
    tokens: Vec<TokenId>,
    req_max_seq_len: usize,
    req_max_new_tokens: usize,
    new_tokens_generated: usize,
    prev_pos: usize,
    no_token_sampling: bool,
    stop_at_end_token: bool,
    sent_stuff_last_time: bool,
    exit_after_one_query: bool,
    result: Vec<u8>, // stores JSONL lines to be returned from read()
}

#[cfg(feature = "server")]
impl GeneratingSession {
    fn read_from_result(&mut self, buf: &mut [u8]) -> usize {
        if !self.result.is_empty() {
            if self.result.len() <= buf.len() {
                for idx in 0..self.result.len() {
                    buf[idx] = self.result[idx];
                }
                let len = self.result.len();
                self.sent_stuff_last_time = true;
                self.result.truncate(0);
                return len;
            } else {
                for idx in 0..buf.len() {
                    buf[idx] = self.result[idx];
                }
                self.result = self.result[buf.len()..].to_vec();
                self.sent_stuff_last_time = true;
                return buf.len();
            }
        }
        return 0;
    }
}

#[cfg(feature = "server")]
impl Read for GeneratingSession {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.sent_stuff_last_time && self.result.is_empty() {
            // If we return WouldBlock every time we send something, it'll cause Rocket to
            // flush available data.
            self.sent_stuff_last_time = false;
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "WouldBlock",
            ));
        }

        // Push more data to the upstream if we have something stored.
        let bytes_read = self.read_from_result(buf);
        if bytes_read > 0 {
            return Ok(bytes_read);
        }
        if self.tokens.len() >= self.req_max_seq_len {
            if self.exit_after_one_query {
                std::process::exit(0);
            }
            return Ok(0);
        }
        if self.new_tokens_generated >= self.req_max_new_tokens {
            if self.exit_after_one_query {
                std::process::exit(0);
            }
            return Ok(0);
        }

        let (mut caches, update_pos) = {
            let mut ac = self.attention_cache_repository.write().unwrap();
            match ac.get(&self.tokens) {
                Some((c, pos)) if pos >= self.prev_pos => (c.true_clone(), pos),
                Some(_) => {
                    std::mem::drop(ac);
                    (self.transformer.make_caches(), 0)
                }
                None => {
                    let caches = self.transformer.make_caches();
                    ac.put(self.tokens.clone(), caches.true_clone(), self.prev_pos);
                    (caches, self.prev_pos)
                }
            }
        };
        if update_pos > self.prev_pos {
            self.prev_pos = update_pos;
        }

        assert!(self.result.is_empty());
        let predictions =
            self.transformer
                .forward(&self.tokens[self.prev_pos..], self.prev_pos, &mut caches);
        self.prev_pos = self.tokens.len();
        let (highest_pred_idx, token_prob) =
            self.token_sampler
                .sample(&predictions, self.tokenizer.as_ref(), &self.tokens);
        self.tokens.push(highest_pred_idx as TokenId);
        {
            let mut ac = self.attention_cache_repository.write().unwrap();
            ac.put(self.tokens.clone(), caches, self.prev_pos);
        }
        self.new_tokens_generated += 1;
        let token: &str = self.tokenizer.id_to_str(highest_pred_idx as TokenId);
        let mut is_end_token: bool = false;
        if token == "</s>" && self.stop_at_end_token {
            self.new_tokens_generated = self.req_max_new_tokens;
            is_end_token = true;
        }

        let mut result: BTreeMap<String, PredResult> = BTreeMap::new();
        if self.no_token_sampling {
            // All predictions go the line.
            let probs = self
                .token_sampler
                .logits_to_btreemap(&predictions, self.tokenizer.as_ref());
            for (k, v) in probs.into_iter() {
                let mut is_end_token: bool = false;
                if k == "</s>" {
                    is_end_token = true;
                }
                result.insert(
                    k,
                    PredResult {
                        p: v,
                        is_end_token: is_end_token,
                    },
                );
            }
            // Convert to JSON
            let json = serde_json::to_string(&result).unwrap();
            self.result.extend(json.as_bytes());
            self.result.push(b'\n');
            return Ok(self.read_from_result(buf));
        } else {
            result.insert(
                token.to_string(),
                PredResult {
                    p: token_prob,
                    is_end_token,
                },
            );
            let json = serde_json::to_string(&result).unwrap();
            self.result.extend(json.as_bytes());
            self.result.push(b'\n');
            return Ok(self.read_from_result(buf));
        }
    }
}

#[cfg(feature = "server")]
struct AttentionCacheRepository {
    caches: BTreeMap<Vec<TokenId>, (TransformerCaches, usize, std::time::Instant)>,
    max_sz: usize,
}

#[cfg(feature = "server")]
impl AttentionCacheRepository {
    fn empty(max_size: usize) -> AttentionCacheRepository {
        AttentionCacheRepository {
            caches: BTreeMap::new(),
            max_sz: max_size,
        }
    }

    /// Makes sure the cache repository is not larger than sz, evicts any older items.
    fn limit_size(&mut self, sz: usize) {
        if sz == 0 {
            self.caches = BTreeMap::new();
            return;
        }
        // Slow algorithm but I guess our cache will never be unimaginably large so it's probably
        // fine
        while self.caches.len() > sz {
            let mut oldest_time = None;
            let mut oldest_key: Option<&Vec<TokenId>> = None;
            for (k, (_, _, time)) in self.caches.iter() {
                if oldest_time.is_none() || time < oldest_time.unwrap() {
                    oldest_time = Some(time);
                    oldest_key = Some(k);
                }
            }
            let oldest_key = oldest_key.unwrap().clone();
            self.caches.remove(&oldest_key);
        }
    }

    fn get(&self, tokens: &[TokenId]) -> Option<(&TransformerCaches, usize)> {
        if let Some((caches, pos, _)) = self.caches.get(tokens) {
            Some((caches, *pos))
        } else {
            None
        }
    }

    fn put(&mut self, tokens: Vec<TokenId>, caches: TransformerCaches, prev_pos: usize) {
        self.caches
            .insert(tokens, (caches, prev_pos, std::time::Instant::now()));
        self.limit_size(self.max_sz);
    }
}

#[cfg(feature = "server")]
#[derive(Clone)]
struct InferenceServerState {
    transformer: Arc<Transformer>,
    tokenizer: Arc<Tokenizer>,
    max_seq_len: usize,
    concurrent_requests_semaphore: Semaphore,
    attention_cache_repository: Arc<RwLock<AttentionCacheRepository>>,
    exit_after_one_query: bool,
}

#[cfg(feature = "server")]
#[post("/", data = "<input>")]
fn handle_request(
    state: State<InferenceServerState>,
    input: Data,
) -> Result<Stream<GeneratingSession>, status::BadRequest<String>> {
    let _lock = state.concurrent_requests_semaphore.acquire();
    let tr = state.transformer.clone();
    let tok = state.tokenizer.clone();

    let mut data = input.open();
    let mut databuf: Vec<u8> = Vec::new();
    data.read_to_end(&mut databuf).unwrap();

    // Parse the JSON out of the request
    let request: InferenceRequest = match serde_json::from_slice(&databuf) {
        Err(_e) => {
            return Err(status::BadRequest(Some("Invalid JSON.".to_string())));
        }
        Ok(ir) => ir,
    };

    let stop_at_end_token = request.stop_at_end_token.unwrap_or(true);
    let temperature = request.temperature.unwrap_or(1.0);
    let top_k = request.top_k.unwrap_or(20);
    let top_p = request.top_p.unwrap_or(1.0);
    let repetition_penalty = request.repetition_penalty.unwrap_or(1.0);
    let mut req_max_seq_len = request.max_seq_len.unwrap_or(state.max_seq_len);
    if req_max_seq_len > state.max_seq_len {
        req_max_seq_len = state.max_seq_len;
    }
    let req_max_new_tokens = request.max_new_tokens.unwrap_or(20);
    let no_token_sampling = request.no_token_sampling.unwrap_or(false);
    let prompt = request.prompt;

    if temperature.is_nan() {
        return Err(status::BadRequest(Some(
            "Temperature must be a number.".to_string(),
        )));
    }
    if top_k == 0 {
        return Err(status::BadRequest(Some(
            "Top-k must be greater than 0.".to_string(),
        )));
    }
    if top_p.is_nan() {
        return Err(status::BadRequest(Some(
            "Top-p must be a number.".to_string(),
        )));
    }
    if repetition_penalty.is_nan() {
        return Err(status::BadRequest(Some(
            "Repetition penalty must be a number.".to_string(),
        )));
    }

    let token_sampler = TokenSampler::new()
        .temperature(temperature)
        .top_p(top_p)
        .top_k(top_k)
        .repetition_penalty(repetition_penalty);
    let toks_id: Vec<TokenId> = tok.tokenize_to_ids(prompt.clone());
    let gsession = GeneratingSession {
        transformer: tr,
        tokenizer: tok,
        attention_cache_repository: state.attention_cache_repository.clone(),
        token_sampler: token_sampler,
        tokens: toks_id,
        req_max_seq_len: req_max_seq_len,
        req_max_new_tokens: req_max_new_tokens,
        new_tokens_generated: 0,
        prev_pos: 0,
        no_token_sampling: no_token_sampling,
        stop_at_end_token: stop_at_end_token,
        sent_stuff_last_time: false,
        exit_after_one_query: state.exit_after_one_query,
        result: Vec::new(),
    };

    return Ok(rocket::response::Stream::chunked(gsession, 1024));
}

fn command_line_inference(
    cli: Cli,
    tr: Arc<Transformer>,
    tok: Arc<Tokenizer>,
    prompt: String,
    interactive_stop: Vec<String>,
    interactive_system_prompt: String,
    interactive_prompt_prefix: String,
    interactive_prompt_postfix: String,
    start_interactive: bool,
    be_quiet: bool,
    max_seq_len: usize,
    params: ModelParams,
    max_threads: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Custom println-like macro that respects be_quiet
    macro_rules! pln {
        ($($arg:tt)*) => {
            if !be_quiet {
                std::println!($($arg)*);
            }
        };
    }

    let mut prompt = prompt;

    if start_interactive && !prompt.is_empty() {
        return Err(
            "Cannot start interactive mode with a prompt. Use --interactive-system-prompt instead."
                .into(),
        );
    }
    prompt = interactive_system_prompt.clone();

    let mut toks_id: Vec<TokenId> = tok.tokenize_to_ids(prompt.clone());
    let mut toks_str: String = prompt.clone();
    let mut prev_pos = 0;
    let mut token_sampler = TokenSampler::new()
        .temperature(1.0)
        .top_p(1.0)
        .top_k(20)
        .repetition_penalty(1.0);

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
    if start_interactive {
        pln!(
            "  Interactive mode stop token sequences: {:?}",
            interactive_stop
        );
        pln!("---");
        pln!("System prompt:");
        pln!("  {}", interactive_system_prompt);
        pln!("---");
        pln!("Interactive prompt prefix: {}", interactive_prompt_prefix);
        pln!("Interactive prompt postfix: {}", interactive_prompt_postfix);
    }
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
    let mut interactive = start_interactive;
    let mut user_token: Vec<TokenId> = vec![];
    while toks_id.len() < max_seq_len {
        let now = std::time::Instant::now();
        let preds = tr.forward(&toks_id[prev_pos..], prev_pos, &mut caches);
        if interactive {
            let mut newinput = String::new();
            std::io::stdin().read_line(&mut newinput)?;
            // removing new line from input
            if newinput.ends_with('\n') {
                let _ = newinput.pop();
            }
            newinput = interactive_prompt_prefix.clone() + &newinput;
            newinput += &interactive_prompt_postfix;
            user_token = tok.tokenize_to_ids(newinput.clone());

            // removing [start token] as it is already in the prompt, and tokenize_to_ids  adds it.
            let _ = user_token.remove(0);
            interactive = false;
        }
        let (highest_pred_idx, token_prob);

        if user_token.len() > 0 {
            highest_pred_idx = user_token.remove(0);
            token_prob = 0.0;
        } else {
            (highest_pred_idx, token_prob) = token_sampler.sample(&preds, &tok, &toks_id);
        }
        toks_id.push(highest_pred_idx as TokenId);

        for (tok_idx, tok_id) in toks_id[prev_pos + 1..].iter().enumerate() {
            if *tok_id == 1 {
                continue;
            }
            let mut tok_print: String = "".to_string();
            let tok_str = tok.id_to_str(*tok_id);
            if tok_str == "</s>" {
                tok_print += "";
                stop_seen = true;
            }
            if tok_str == "<0x0A>" {
                tok_print += "\n";
            } else {
                tok_print += tok_str.replace('▁', " ").as_str();
            }
            toks_str += tok_print.as_str();
            if first && tok_idx < toks_id.len() - 2 {
                // intentionally left empty, already print
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
                    tok_print.truecolor(128 + redness / 2, 255 - redness / 2, 128)
                );
            };
            for stop_str in interactive_stop.iter() {
                if !first && toks_str.ends_with(stop_str.as_str()) {
                    if start_interactive {
                        interactive = true;
                    }
                    break;
                }
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
        if times_per_token.len() > 0 {
            println!(
                "Time taken per token (excluding first token): {:?}ms",
                times_per_token.iter().map(|t| t.as_millis()).sum::<u128>()
                    / times_per_token.len() as u128
            );
        } else {
            println!("No token generated");
        }
    }
    Ok(())
}
