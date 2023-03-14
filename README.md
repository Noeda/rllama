# RLLaMA

This is my attempt at making the LLaMA language model working on a pure Rust
CPU implementation. I was inspired by an amazing CPU implementation here:
https://github.com/ggerganov/ggml that could run GPT-J 8B models.

As of writing of this, this can run LLaMA-7B at around ~1 token per second, on
a Ryzen 3950X using something like 1.5 threads because I haven't yet properly
figured out how to multithread this.

I've also managed to run LLaMA-13B which just barely fits in my 64-gig machine
with 32-bit float weights everywhere.

LLaMA-30B technically runs but my computer does not have enough memory to keep
all the weights around so generating a token takes minutes.

I have not tried LLaMA-60B but presumably if all the smaller models work it
would run given a sufficiently chonky computer.

This uses AVX2 intrinsics to speed up itself. Therefore, you need an x86-family
CPU to run this.

It also has a Python unpickler that understands the `.pth` files used by
PyTorch. Well almost, it doesn't unzip them automatically (see below).

# How to run

You will need Rust. Make sure you can run `cargo` from a command line. In
particular, this is using unstable features so you need nightly rust. Make sure
if you write `cargo --version` it is nightly.

You will need to download LLaMA-7B weights. Refer to https://github.com/facebookresearch/llama/

Once you have 7B weights, and the `tokenizer.model` it comes with, you need to
decompress it.

```shell
$ cd LLaMA
$ cd 7B
$ unzip consolidated.00.pth
# For LLaMA-7B, rename consolidated to consolidated.00
# For the larger models, the number is there already so no need to do this step.
$ mv consolidated consolidated.00
```

You should then be ready to generate some text.

```shell
cargo run --release -- --tokenizer-model /path/to/tokenizer.model --model-path /path/to/LLaMA/7B --param-path /path/to/LLaMA/7B/params.json --prompt "The meaning of life is"
```

Right now it seems to use around ~25 gigabytes of memory for 7B and around ~50
gigabytes for 13B. If you don't use OpenCL, then all parameters are cast to
32-bit floats.

You can use `--temperature`, `--top-p` and `--top-k` to adjust token sampler
settings.

# Future plans

This is a hobby thing for me so don't expect updates or help.

* Some other CPU implementations use quantization to reduce the size of weights
* Put some of the operations on the OpenCL GPU/CPU. I've made some initial
  OpenCL code but there's still bunch of stuff that could be OpenCLified.
  The OpenCL code is fast for both GPU OpenCL and CPU OpenCL (better than my
  own handwritten AVX2 code which makes me sad).
* I've heard there is some thing called Tensor Cores on NVidia GPUs. Not
  accessible with OpenCL. But might be accessible on Vulkan with a an
  extension.
* More sophisticated token sampling. I saw on Hackernews some comments how the
  samplers are kinda garbage and you can get much better results with good
  defaults and things like repetition penalty.
* There is an initial start-up time as the program has to pass through the
  initial prompt. I don't know if this start-up time can be eliminated
  completely but it could be cached on disk. Use cases like having a standard
  prompt to prime the text generation that you reuse many times.

# Benchmarks

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
LLaMA-7B:  AMD Ryzen 3950X + OpenCL GTX 3090 Ti:  567ms / token
LLaMA-7B:  AMD Ryzen 3950X + OpenCL Ryzen 3950X:  956ms / token
LLaMA-13B: AMD Ryzen 3950X + OpenCL GTX 3090 Ti:  987ms / token
LLaMA-13B: AMD Ryzen 3950X + OpenCL Ryzen 3950X:  1706ms / token

# commit 35b0c372a87192761e17beb421699ea5ad4ac1ce  (13 March 2023)
# I moved some attention stuff to OpenCL too.
LLaMA-7B:  AMD Ryzen 3950X + OpenCL GTX 3090 Ti:  283ms / token
LLaMA-7B:  AMD Ryzen 3950X + OpenCL Ryzen 3950X:  679ms / token
LLaMA-13B: AMD Ryzen 3950X + OpenCL GTX 3090 Ti:  <ran out of GPU memory>
LLaMA-13B: AMD Ryzen 3950X + OpenCL Ryzen 3950X:  1226ms / token
```
