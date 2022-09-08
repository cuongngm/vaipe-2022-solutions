FROM nvidia/cuda:11.2.1-runtime-ubuntu20.04
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get -y install python3.8
RUN apt-get -y install python3-pip
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update
RUN apt-get -y install git wget vim
RUN pip install --upgrade pip
RUN mkdir /app
ADD . /app
WORKDIR /app
RUN bash scripts/install.sh
