# FROM python:3.11.13-slim-bullseye

# ADD ./evaluation /evaluation
# RUN mkdir /dataset
# COPY ./pose_json /dataset
# WORKDIR /evaluation

# RUN apt-get update && apt-get install -y && apt-get install g++ -y && apt-get install gcc -y
# RUN apt-get install -y git wget unzip

# RUN pip3 install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
# RUN pip3 install -U openmim
# RUN mim install mmengine
# RUN mim install mmcv
# RUN git clone https://github.com/open-mmlab/mmpose.git
# CMD [ "cd mmpose" ]
# ENTRYPOINT ["bash"]

#----------------------------------------------------------------------------------------------------------------------------
# This Dockerfile is for building a container with MMPose installed, using PyTorch and CUDA.

# ARG PYTORCH="1.8.1"
# ARG CUDA="10.2"
# ARG CUDNN="7"

# FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
# ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
# ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# # To fix GPG key error when running apt-get update
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx\
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Install xtcocotools
# RUN pip install cython
# RUN pip install xtcocotools

# # Install MMEngine and MMCV
# RUN pip install openmim
# RUN mim install mmengine "mmcv>=2.0.0"

# # Install MMPose
# RUN conda clean --all
# RUN git clone https://github.com/open-mmlab/mmpose.git /mmpose
# WORKDIR /mmpose
# RUN git checkout main
# ENV FORCE_CUDA="1"
# RUN pip install -r requirements/build.txt
# RUN pip install --no-cache-dir -e .


#-----------------------------------------------------------------------------------------------------------------------------
FROM python:3.11.13-slim-bullseye

# ADD ./evaluation /evaluation
# RUN mkdir /dataset
# COPY ./pose_json /custom_dataset

RUN apt-get update && apt-get install -y
RUN apt-get install -y git wget unzip ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx g++ gcc\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
RUN pip install -U openmim
RUN mim install mmengine

RUN pip install cython
RUN pip install git+https://github.com/jin-s13/xtcocoapi

RUN pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
RUN mim install mmdet

RUN git clone https://github.com/open-mmlab/mmpose.git
WORKDIR /mmpose
RUN git checkout main
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -e .