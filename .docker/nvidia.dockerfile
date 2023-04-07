FROM debian:bookworm

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update -y
RUN apt install -y curl \
    apt-utils \
    unzip \
    tar \
    curl \
    xz-utils \
    ocl-icd-libopencl1 \
    opencl-headers \
    clinfo \
    build-essential \
    gcc

RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > /rustup.sh
RUN chmod +x /rustup.sh
RUN /rustup.sh -y

RUN apt install -y opencl-dev

RUN bash -c 'export LD_LIBRARY_PATH=/usr/lib:/lib:/usr/lib64:/lib64; export PATH="$PATH:$HOME/.cargo/bin";rustup default nightly'

COPY . /opt/rllama
RUN bash -c 'export PATH="$PATH:$HOME/.cargo/bin";cd /opt/rllama;RUSTFLAGS="-C target-feature=+sse2,+avx,+fma,+avx2" cargo build --release --features server'
RUN ln -s /opt/rllama/target/release/rllama /usr/bin
