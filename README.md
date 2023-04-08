# RLLaMA

RLLaMA is a pure Rust implementation of [LLaMA large language model inference.](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/).

## Supported features

  * Uses either `f16` and `f32` weights.
  * LLaMA-7B, LLaMA-13B, LLaMA-30B, LLaMA-65B all confirmed working
  * Hand-optimized AVX2 implementation
  * OpenCL support for GPU inference.
  * Load model only partially to GPU with `--percentage-to-gpu` command line switch to run hybrid-GPU-CPU inference.
  * Simple HTTP API support, with the possibility of doing token sampling on
    client side
  * It can load `Vicuna-13B` instruct-finetuned model (although currently there is no nice UX).

## Performance

The current performance is as follows:

```
Pure Rust implementations:

LLaMA-7B:  AMD Ryzen 3950X:                       552ms / token     f16    (pure Rust)
LLaMA-7B:  AMD Ryzen 3950X:                       1008ms / token    f32    (pure Rust)
LLaMA-13B: AMD Ryzen 3950X:                       1029ms / token    f16    (pure Rust)
LLaMA-13B: AMD Ryzen 3950X:                       1930ms / token    f32    (pure Rust)
LLaMA-30B: AMD Ryzen 5950X:                       2112ms / token    f16    (pure Rust)
LLaMA-65B: AMD Ryzen 5950X:                       4186ms / token    f16    (pure Rust)

OpenCL (all use f16):

LLaMA-7B:  AMD Ryzen 3950X + OpenCL RTX 3090 Ti:  216ms / token            (OpenCL on GPU)
LLaMA-7B:  AMD Ryzen 3950X + OpenCL Ryzen 3950X:  680ms / token            (OpenCL on CPU)
LLaMA-13B: AMD Ryzen 3950X + OpenCL RTX 3090 Ti:  420ms / token            (OpenCL on GPU)
LLaMA-13B: AMD Ryzen 3950X + OpenCL Ryzen 3950X:  1232ms / token           (OpenCL on CPU)
LLaMA-30B: AMD Ryzen 5950X + OpenCL Ryzen 5950X:  4098ms / token           (OpenCL on CPU)
```

Scroll to the bottom of this README.md to see benchmarks over time.

## Screenshot

![Screenshot of RLLaMA in action](rllama.gif)

## Install

You can install with `cargo` tool. RLLaMA uses intrinsics extensively and you
likely need to enable them to install the executable.

```
RUSTFLAGS="-C target-feature=+sse2,+avx,+fma,+avx2" cargo install rllama
```

There is a `.cargo/config.toml` inside this repository that will enable these
features if you install manually from this Git repository instead.

## Install (Docker path)

There is a Dockerfile you can use if you'd rather just get started quickly and
you are familiar with `docker`. You still need to download the models yourself.

```
docker build -f ./.docker/cpu.dockerfile -t rllama .
```

```
docker run -v /models/LLaMA:/models:z -it rllama \
    rllama --model-path /models/7B \
           --param-path /models/7B/params.json \
           --tokenizer-path /models/tokenizer.model \
           --prompt "hi I like cheese"
```

Replace `/models/LLaMA` with the directory you've downloaded your models to.
The `:z` in `-v` flag may or may not be needed depending on your distribution
(I needed it on Fedora Linux)

## LLaMA weights

Refer to https://github.com/facebookresearch/llama/ As of now, you need to be
approved to get weights.

For LLaMA-7B make sure, you got these files:

```shell
* 7B/consolidated.00.pth
* 7B/params.json
* tokenizer.model
```

The `consolidated.00.pth` is actually a zip file. You need to unzip it:

```shell
$ cd 7B
$ unzip consolidated.00.pth
$ mv consolidated consolidated.00
```

If you are using a larger model like LLaMA-13B, then you can skip the last step
of renaming the `consolidated` directory.

You should now be ready to generate some text.

## Example

Run LLaMA-7B with some weights casted to 16-bit floats:

```shell
rllama --tokenizer-path /path/to/tokenizer.model \
       --model-path /path/to/LLaMA/7B \
       --param-path /path/to/LLaMA/7B/params.json \
       --f16 \
       --prompt "The meaning of life is"
```

Use `rllama --help` to see all the options.

## Partially load model to GPU

`rllama` can load only some of the transformer blocks to GPU. There is a
command line argument:

`--percentage-to-gpu <value between 0 and 1, defaults to 1>`

1 means 100% and 0 means 0%. Values in-between load the model partially to GPU.

You can use this to load LLaMA-13B or Vicuna-13B on a consumer GPU of 24
gigabytes at around `--percentage-to-gpu 0.9` before it fails to out-of-memory
error (if there are no competing programs on the computer that use GPU memory).

## Interactive mode

There is a simple experimental interactive mode to try force a type of
back-and-forth discussion with the model.

```shell
rllama ... --start-interactive \
           --interactive-system-prompt "Helpful assistant helps curious human." \   # (optional)
           --interactive-prompt-postfix  " ###Assistant:" \  # (optional)
           --interactive-stop "###Human: "                   # (optional)
```

In this mode, you need to type your prompt before the AI starts doing its work.
If the AI outputs token sequence given in `--interactive-stop` (defaults to
`###Human:`) then it will ask for another input.

The defaults match Vicuna-13B model:

```
  --interactive-system-prompt    "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
  --interactive-prompt-postfix   " ###Assissant:"
  --interactive-prompt-prefix    " "
  --interactive-stop             "###Human:"
```

`--interactive-prompt-postfix` is appended automatically to your typed text and
`--interactive-prompt-prefix` is appended to the start of your typed text.Here
is an example of interactive mode command line with the default settings:

```shell
rllama --f16 \
       --param-path /models/vicuna13b/params.json \
       --model-path /models/vicuna13b \
       --tokenizer-path /stonks/LLaMA/tokenizer.model \
       --start-interactive
```

As of writing of this, the output is not formatted prettily for chat and there
is no visual indication of when you are supposed to be typing. That will come
later.

## Inference server

`rllama` can run in an inference server mode with a simple HTTP JSON API. You
need to enable `server` features for this.

```
cargo build --release --features server
```

The command line flags for this are:

  * `--inference-server` using this will turn on the inference server.
  * `--inference-server-port` sets the port. Default port is 8080.
  * `--inference-server-host` sets the host. The default host is 127.0.0.1.
  * `--inference-server-max-concurrent-inferences` sets how many concurrent
    requests are allowed to be actively doing inference at the same time. The
    default is 5.
  * `--inference-server-api-path` sets which path servers the API requests. The
    default path is `/rllama/v1/inference`
  * `--inference-server-prompt-cache-size` sets how many previous prompt
    calculations should be cached. Default is 50. This speeds up token
    generation for prompts that were already requested before, however it also
    increases memory use as the cache gets more full.
  * `--inference-server-exit-after-one-query` will make the server exit with
    exit code 0 after it has served one HTTP query. This is used for
    troubleshooting and experiments.

Prompts and flags related to token sampling are all ignored in inference server
mode. Instead, they are obtained from each HTTP JSON API request.

### Inference server API

There is an `examples/api_hello_world.py` for a minimal API use example.

```
POST /rllama/v1/inference
```

Expects a JSON body and `Accept: application/json` or `Accept: text/jsonl`.

The expected JSON is as follows:

```
  {
     "temperature":        <number, optional>
     "top_k":              <integer, optional, default 20>
     "top_p":              <number, optional, default: 1.0>
     "repetition_penalty": <number, optional, default: 1.0>
     "stop_at_end_token":  <bool, optional, default: true>
     "max_seq_len":        <integer, optional, default: 1024. Clamped to
                            be at highest the same as --max-seq-len command line option.>
     "max_new_tokens":     <integer, optional, default: 1024>
     "no_token_sampling":  <bool, optional, default: false>
     "prompt":             <string, required>
  }
```

The form of the response depends on if `no_token_sampling` is set to true or false. The
response is in JSONL, i.e. multiple JSON dictionaries, separated by newlines.

`no_token_sampling` can turn off `rllama`'s own token sampling. In this case,
the probabilities for every token are returned instead.

When no\_token\_sampling = false:

```
{<token string>: {"p": <number>, "is_end_token": bool, might not be present}}
```

  * `token` contains the new token to be appended to output. It does not
    include string you fed to the system originally.
  * `p` is the probability that this token was chosen. For example, if this
    value is 0.1, it means that this particular token had 10% chance of being
    selected with the current token sampling settings.
  * `is_end_token` is `true` is the given token signifies end of output. This
    field is not present otherwise.

When no\_token\_sampling = true:

```
{<token string>: {"p": <number>, "is_end_token": bool, might not be present} \
,<token string>: {"p": <number>, "is_end_token": bool, might not be present} \
,...}
```

If you want to implement your own token sampling, you may want to set
`max_new_tokens=1` and `stop_at_end_token=false` to suppress rllama's own
sampling behavior entirely.

`rllama` internally caches recently queried prompts and the intermediate
computations so that it's able to continue off quickly if you issue a query
that is either the same as a previous query or a continuation of one.

## How to turn on OpenCL

Use `opencl` Cargo feature.

```
RUSTFLAGS="-C target-feature=+sse2,+avx,+fma,+avx2" cargo install rllama --features opencl
```

```
rllama --tokenizer-path /path/to/tokenizer.model \
       --model-path /path/to/LLaMA/7B \
       --param-path /path/to/LLaMA/7B/params.json \
       --opencl-device 0 \
       --prompt "The meaning of life is"
```

With `opencl` feature, there is also another argument, `--opencl-device` that
takes a number. That number selects Nth OpenCL device found on the system. You
can see the devices in the output when you run the program (e.g. see the
screenshot below).

Weights are always cast to 16-bit floats for OpenCL.

## Notes and future plans

This is a hobby thing for me so don't expect updates or help.

* There are various BLAS libraries like CLBlast to speed up matrix
  multiplication that probably outperform my handwritten code.
* I've heard there is some thing called Tensor Cores on nVidia GPUs. Not
  accessible with OpenCL. But might be accessible on Vulkan with a an
  extension. Or with cuBLAS.

## Benchmarks

I'm trying to track that I'm making this faster and not slower.

For 50-length sequence generation:

```
cargo run --release --
          --model-path /LLaMA/13B \
          --param-path /LLaMA/13B/params.json \
          --tokenizer-path /LLaMA/tokenizer.model \
          --prompt "Computers are pretty complica" --max-seq-len 50

# commit c9c861d199bd2d87d7e883e3087661c1e287f6c4  (13 March 2023)

LLaMA-7B:  AMD Ryzen 3950X: 1058ms / token
LLaMA-13B: AMD Ryzen 3950X: 2005ms / token

# commit 63d27dba9091823f8ba11a270ab5790d6f597311  (13 March 2023)
# This one has one part of the transformer moved to GPU as a type of smoke test

LLaMA-7B:  AMD Ryzen 3950X + OpenCL RTX 3090 Ti:  567ms / token
LLaMA-7B:  AMD Ryzen 3950X + OpenCL Ryzen 3950X:  956ms / token
LLaMA-13B: AMD Ryzen 3950X + OpenCL RTX 3090 Ti:  987ms / token
LLaMA-13B: AMD Ryzen 3950X + OpenCL Ryzen 3950X:  1706ms / token

# commit 35b0c372a87192761e17beb421699ea5ad4ac1ce  (13 March 2023)
# I moved some attention stuff to OpenCL too.

LLaMA-7B:  AMD Ryzen 3950X + OpenCL RTX 3090 Ti:  283ms / token
LLaMA-7B:  AMD Ryzen 3950X + OpenCL Ryzen 3950X:  679ms / token
LLaMA-13B: AMD Ryzen 3950X + OpenCL RTX 3090 Ti:  <ran out of GPU memory>
LLaMA-13B: AMD Ryzen 3950X + OpenCL Ryzen 3950X:  1226ms / token

# commit de5dd592777b3a4f5a9e8c93c8aeef25b9294364  (15 March 2023)
# The matrix multiplication on GPU is now much faster. It didn't have that much
# effect overall though, but I got modest improvement on LLaMA-7B GPU.

LLaMA-7B:  AMD Ryzen 3950X + OpenCL RTX 3090 Ti:  247ms / token
LLaMA-7B:  AMD Ryzen 3950X + OpenCL Ryzen 3950X:  680ms / token
LLaMA-13B: AMD Ryzen 3950X + OpenCL RTX 3090 Ti:  <ran out of GPU memory>
LLaMA-13B: AMD Ryzen 3950X + OpenCL Ryzen 3950X:  1232ms / token
LLaMA-30B: AMD Ryzen 5950X + OpenCL Ryzen 5950X:  4098ms / token

# commit 3d0afcf24309f28ec540ed7645c35400a865ad6f  (17 March 2023)
# I've been focusing on making the ordinary non-OpenCL CPU implementation
# faster and I got some gains, most importantly from multithreading.
# There is Float16 support now, so I've added f16/f32 to these tables:
#
# I also managed to run LLaMA-65B for the first time.

LLaMA-7B:  AMD Ryzen 3950X: 552ms / token     f16
LLaMA-7B:  AMD Ryzen 3950X: 1008ms / token    f32
LLaMA-13B: AMD Ryzen 3950X: 1029ms / token    f16
LLaMA-13B: AMD Ryzen 3950X: 1930ms / token    f32
LLaMA-30B: AMD Ryzen 5950X: 2112ms / token    f16
LLaMA-65B: AMD Ryzen 5950X: 4186ms / token    f16

# commit f5328ab5bd62fe9bd930539382b13e9033434a0b (5 April 2023)
# I've worked on making Vicuna-13B runnable and added an option to only
# partially use GPU. Improved one of the OpenCL kernels:

LLaMA-7B:   AMD Ryzen 3950X + OpenCL RTX 3090 Ti:    420ms (at 90%/10% GPU/CPU split)
LLaMA-13B:  AMD Ryzen 3950X + OpenCL RTX 3090 Ti:    216ms (at 100% GPU)
```
