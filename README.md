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
gigabytes for 13B. Internally all weights are cast to 32-bit floats.

You can use `--temperature`, `--top-p` and `--top-k` to adjust token sampler
settings.

# Future plans

This is a hobby thing for me so don't expect updates or help.

* Some other CPU implementations use quantization to reduce the size of weights
* Put some of the operations on the OpenCL GPU/CPU. I've made some initial
  OpenCL code but it is not used in the transformer loop yet. The CPU OpenCL
  improves my own AVX2 code by like 100% and massively so on GPU although I am
  also like 20x slower than equivalent operation on PyTorch on the same GPU.
* I've heard there is some thing called Tensor Cores on nVidia GPUs. Not
  accessible with OpenCL. But might be accessible on Vulkan with a an
  extension.
