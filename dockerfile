FROM python:3.11.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update 
RUN apt-get install -y git
RUN apt-get install -y libgtk2.0-dev
RUN apt-get install -y libgl1-mesa-glx
RUN rm -rf /var/lib/apt/lists/* 

WORKDIR /root