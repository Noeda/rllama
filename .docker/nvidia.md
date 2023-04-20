#rllama docker on nvidia

## Getting OpenCL to work inside docker.
Please note that this also requires some packages and modifications on your host system in order to allow the containers to use nvidia GPU features such as **compute**.


For each of the described distro / distro-family you could follow the instructions at the given links below.

**Note**: You also need an upto-date version of docker/docker-ce so be sure to follow the instructions to install docker for your distro from the [docker website](https://docs.docker.com/engine/install).

**Note2**: I have only personally tested the instructions on fedora/nobara and hence, cannot guarantee the accuracy of the instructions for other distros.

### Fedora / Fedora-based
**[https://gist.github.com/JuanM04/fcbed16d0f4405a286adebee5fd31cb2](https://gist.github.com/JuanM04/fcbed16d0f4405a286adebee5fd31cb2)**


### Debian / Debian-based / Ubuntu / Ubuntu-based
**[https://www.howtogeek.com/devops/how-to-use-an-nvidia-gpu-with-docker-containers/](https://www.howtogeek.com/devops/how-to-use-an-nvidia-gpu-with-docker-containers/)**


### Arch / Arch-based
**[https://wiki.archlinux.org/title/Docker#Run_GPU_accelerated_Docker_containers_with_NVIDIA_GPUs](https://wiki.archlinux.org/title/Docker#Run_GPU_accelerated_Docker_containers_with_NVIDIA_GPUs)**

Feel free to contribute/improve the instructions for existing and other distros.

## Usage
1. 
```bash
docker build -f ./.docker/nvidia.dockerfile -t rllama:nvidia .
```
2.
```bash
docker run --rm --gpus all --privileged -v /models/LLaMA:/models:z -it rllama:nvidia \
    rllama --model-path /models/7B \
           --param-path /models/7B/params.json \
           --tokenizer-path /models/tokenizer.model \
           --prompt "hi I like cheese"
```

Replace `/models/LLaMA` with the directory you've downloaded your models to. The `:z` in `-v` flag may or may not be needed depending on your distribution (I needed it on Fedora Linux)