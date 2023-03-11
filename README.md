# AdeonLLaMA

This is my attempt at making the LLaMA language model working on a pure Rust
CPU implementation. I was inspired by an amazing CPU implementation here:
https://github.com/ggerganov/ggml that could run GPT-J 8B models.

As of writing of this, this can run LLaMA-7B at around ~1 token per second, on
a Ryzen 3950X using something like 1.5 threads because I haven't yet properly
figured out how to multithread this.

It uses AVX2 intrinsics to speed up itself. Therefore, you need an x86-family
CPU to run this.

It has a Python unpickler that understands the `.pth` files used by PyTorch.
Well sort of, it doesn't unzip them automatically (see below).

# How to run

You will need Rust. Make sure you can run `cargo` from a command line.

You will need to download LLaMA-7B weights. Refer to https://github.com/facebookresearch/llama/

Once you have 7B weights, and the `tokenizer.model` it comes with, you need to
decompress it.

```shell
$ cd LLaMA
$ cd 7B
$ unzip consolidated.00.pth
```

You should then be ready to generate some text.

```shell
cargo run --release -- --tokenizer-model /path/to/tokenizer.model --model-path /path/to/LLaMA/7B/consolidated/data.pkl --prompt "The meaning of life is"
```

Right now it seems to use around ~25 gigabytes of memory. Internally all
weights are cast to 32-bit floats.

You can use `--temperature`, `--top-p` and `--top-k` to adjust token sampler
settings.

# Future plans

This is a hobby thing for me so don't expect updates or help.
