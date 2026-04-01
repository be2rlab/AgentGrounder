FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

SHELL [ "/bin/bash", "-c" ]
ENV DEBIAN_FRONTEND noninteractive
ENV USER=docker_user

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo git wget nano unzip ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR /opt/conda
ENV PATH=${CONDA_DIR}/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r


RUN conda create -n pytorch3d python=3.12
RUN conda install -y -n pytorch3d cuda-toolkit=12.9 -c nvidia
RUN conda run -n pytorch3d pip install torch==2.9.1+cu129 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129


RUN conda run -n pytorch3d pip install iopath fvcore
RUN conda install -y -n pytorch3d iopath -c iopath -c conda-forge

#### PyTorch3D
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
RUN conda run -n pytorch3d pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"

WORKDIR /home/${USER}

RUN conda run -n pytorch3d pip install open3d
RUN conda run -n pytorch3d pip install jsonlines
RUN apt-get update && apt-get install libgl1-mesa-glx -y
RUN conda run -n pytorch3d pip install openai

RUN conda run -n pytorch3d pip install opencv-python nltk
RUN conda run -n pytorch3d pip install "numpy<2.0"
RUN apt-get update && apt-get install -y fonts-freefont-ttf fonts-dejavu-core
RUN conda run -n pytorch3d pip install ollama
RUN conda run -n pytorch3d pip install langchain langchain-ollama langchain-community
RUN conda run -n pytorch3d pip install point-cloud-utils

#### SAM3
RUN git clone https://github.com/facebookresearch/sam3.git && cd sam3 && \
    git checkout 2ec3c0711a9719ac324388adc662e471f8fe2168 && \
    conda run -n pytorch3d pip install -e .

RUN conda run -n pytorch3d pip install einops triton decord pycocotools psutil
RUN conda run -n pytorch3d pip install matplotlib


#### Rex Omni
RUN git clone https://github.com/IDEA-Research/Rex-Omni.git && cd Rex-Omni && \
    git checkout 8c474a32fd9a9064742a8bb87695a751d70b39b5 && \
    conda run -n pytorch3d pip install  --no-deps -e .

RUN conda run -n pytorch3d pip install qwen_vl_utils transformers accelerate
RUN conda run -n pytorch3d pip install pydantic gradio gradio_image_prompter
RUN conda run -n pytorch3d pip install vllm # --no-build-isolation
RUN conda run -n pytorch3d pip install flash-attn --no-build-isolation # ==2.7.4.post1

#### Ultralytics
RUN conda run -n pytorch3d pip install ultralytics
RUN conda run -n pytorch3d pip install git+https://github.com/ultralytics/CLIP.git

#### Extra
RUN conda run -n pytorch3d pip install pybboxes --no-deps
RUN conda run -n pytorch3d pip install langchain_chroma

ENV LD_LIBRARY_PATH=/opt/conda/envs/pytorch3d/lib:$LD_LIBRARY_PATH

ARG UID=1000
ARG GID=1000
RUN useradd -m ${USER} --uid=${UID} \
    && usermod -s /bin/bash ${USER} \
    && usermod -a -G sudo ${USER} \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
WORKDIR /home/${USER}/PCGrounder
RUN chown -R $USER /home/${USER}
USER ${USER}

RUN echo 'source activate pytorch3d' >> /home/${USER}/.bashrc