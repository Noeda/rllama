#[cfg(not(target_feature = "avx2"))]
compile_error!("This library assumes availability of AVX and must be compiled with -C target-feature=+sse2,+avx,+fma,+avx2");
#[cfg(not(target_feature = "sse2"))]
compile_error!("This library assumes availability of AVX and must be compiled with -C target-feature=+sse2,+avx,+fma,+avx2");
#[cfg(not(target_feature = "fma"))]
compile_error!("This library assumes availability of AVX and must be compiled with -C target-feature=+sse2,+avx,+fma,+avx2");
#[cfg(not(target_feature = "avx"))]
compile_error!("This library assumes availability of AVX and must be compiled with -C target-feature=+sse2,+avx,+fma,+avx2");

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    rllama::rllama_main::main()
}
