
ARG TENSORFLOW="1.13.2"

FROM tensorflow/tensorflow:${TENSORFLOW}-gpu-py3-jupyter


ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git \
	ninja-build \
	libglib2.0-0 \
	libsm6 \
	libxrender-dev \
	libxext6 \
	libgl1-mesa-glx \
	python3-pip \
	vim \
	libstdc++6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN  pip install --upgrade pip

RUN pip3 install \
    # numpy==1.16.2 \ # ovajdaje gresku kada loadas load_graph_mtx
    numpy==1.16.1 \
    scikit-learn \
    scipy \
    PyYAML \
    plotly \
    opencv-python \
    configparser \
    matplotlib \
    ConfigArgParse \
    tqdm \
    gdown \
    trimesh>=3.6 \
#    tensorflow-gpu==1.13.2 \
    torch==1.2.0 \
    smplx==0.1.13 \
    https://github.com/MPI-IS/mesh/releases/download/v0.3/psbody_mesh-0.3-cp36-cp36m-linux_x86_64.whl

#RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
#    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - \
#    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list |  tee /etc/apt/sources.list.d/nvidia-docker.list


#RUN apt-get update && apt-get install -y nvidia-docker2

#RUN export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

WORKDIR /
