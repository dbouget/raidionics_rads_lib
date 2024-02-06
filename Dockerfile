FROM python:3.8-slim

MAINTAINER David Bouget <david.bouget@sintef.no>

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get -y install sudo
RUN apt-get update && apt-get install -y git

WORKDIR /workspace

RUN git clone https://github.com/dbouget/raidionics_rads_lib.git
RUN pip3 install --upgrade pip
RUN pip3 install -e raidionics_rads_lib

RUN mkdir /workspace/resources

ENTRYPOINT ["python3","/workspace/raidionics_rads_lib/main.py"]