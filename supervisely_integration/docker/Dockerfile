FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install python3-pip -y
RUN apt-get install -y git
RUN python3 -m pip install --upgrade pip

RUN pip3 install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
RUN git clone https://github.com/facebookresearch/co-tracker
WORKDIR co-tracker

RUN pip3 install -e .

WORKDIR /
RUN pip3 install opencv-python einops timm matplotlib moviepy flow_vis

RUN python3 -m pip install supervisely==6.73.516
RUN apt-get update
RUN apt-get install ffmpeg libgeos-dev libsm6 libxext6 libexiv2-dev libxrender-dev libboost-all-dev -y

RUN apt-get -y install wget
RUN mkdir -p checkpoints

WORKDIR checkpoints

RUN wget https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_4_wind_8.pth
RUN wget https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_4_wind_12.pth
RUN wget https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_8_wind_16.pth

LABEL python_sdk_version="6.73.486"