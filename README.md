# AdeonLLaMA

This is my attempt at making the LLaMA language model working on a pure Rust
CPU implementation.

As of writing of this, it can run LLaMA-7B at around ~1 token per second, using
something like 1.5 threads because I haven't yet properly figured out how to
multithread this.

It uses AVX2 intrinsics to speed up itself.

# How to run

You will need the LLaMA-7B weights first. Refer to https://github.com/facebookresearch/llama/

Once you have 7B weights, and the `tokenizer.model` it comes with, you can make
it generate tokens:

```shell
cargo run --release -- --tokenizer-model /path/to/tokenizer.model --model-path /path/to/LLaMA/7B
```
