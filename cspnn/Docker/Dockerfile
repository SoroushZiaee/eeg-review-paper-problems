# FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04
#nvidia/cuda:12.3.1-devel-ubuntu20.04
FROM ubuntu:22.04

MAINTAINER Arman Ali Mohammadi

# fix nvidia cuda image symlinc problem
# RUN cd /usr/local/cuda/lib64 && \
# 	ln -s libnvrtc.so.11.2 libnvrtc.so && \
# 	ln -s libnvrtc-builtins.so.11.4 libnvrtc-builtins.so

# ENV CUDA_PATH="/usr/local/cuda"
# ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"

ENV DEBIAN_FRONTEND="noninteractive"

## installing Python 3.8 
## (Note: Comment this section if you use python official image as base image)
RUN apt-get update && \
	apt-get install -y software-properties-common wget curl htop build-essential && \
	add-apt-repository ppa:deadsnakes/ppa && \
	apt-get update && \
	apt-get install -y python3.8 python3.8-dev python3-pip python3.8-distutils && \
	ln -s /usr/bin/python3.8 /usr/bin/python && \
	curl https://bootstrap.pypa.io/get-pip.py | python && \
	pip install --upgrade pip && \
	apt-get purge -y --auto-remove build-essential python3.8-dev && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*
## Done with python installation

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

# install dependencies
COPY ./requirements.txt .
RUN apt-get update && \
	apt-get install -y software-properties-common \
						wget \
						git \
						curl \
						htop \
						build-essential \
						python3.8-dev \
						unzip && \
    # pip install --no-cache-dir -r requirements.txt && \
	# apt-get purge -y --auto-remove git build-essential python3.8-dev && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt


# install profiling tools
# offline packages can be downloaded from:
# https://developer.nvidia.com/gameworksdownload#?dn=nsight-systems-2021-5-1

# only cli (remote node) [recommended]
# COPY ./Docker/NsightSystems-linux-cli-public-2021.5.1.77-4a17e7d.deb .

# RUN git clone --branch r21.04 --depth 1 https://github.com/NVIDIA/PyProf.git && \
# 	cd PyProf && \
# 	pip install --no-cache-dir . && \
# 	cd .. && \
# 	dpkg -i NsightSystems-linux-cli-public-2021.5.1.77-4a17e7d.deb && \
# 	apt install -y -f && \
# 	apt-get clean && \
# 	rm -rf /var/lib/apt/lists/*


# alternative 1
# complete with GUI
# COPY ./Docker/NsightSystems-linux-public-2021.5.1.77-4a17e7d.deb .

# RUN git clone --branch r21.04 --depth 1 https://github.com/NVIDIA/PyProf.git && \
# 	cd PyProf && \
# 	pip install --no-cache-dir . && \
# 	cd .. && \
# 	dpkg -i NsightSystems-linux-public-2021.5.1.77-4a17e7d.deb && \
# 	# wget https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/NsightSystems-linux-public-2021.5.1.77-4a17e7d.deb && \
# 	# apt install -y ./NsightSystems-linux-public-2021.5.1.77-4a17e7d.deb && \
# 	apt install -y -f && \
# 	apt-get clean && \
# 	rm -rf /var/lib/apt/lists/*

# alternative 2
# RUN git clone --branch r21.04 --depth 1 https://github.com/NVIDIA/PyProf.git && \
# 	cd PyProf && \
# 	pip install --no-cache-dir . && \
# 	cd .. && \
# 	apt-key adv --fetch-keys https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/nvidia.pub && \
# 	add-apt-repository -y "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/ /" && \
# 	apt update && \
# 	apt install -y nvidia-insights && \
# 	apt install -y -f && \
# 	apt-get clean && \
# 	rm -rf /var/lib/apt/lists/*


CMD ["bash"]