
# creates virtual ubuntu in docker image
FROM python:3.8-slim

# maintainer of docker file
MAINTAINER David Bouget <david.bouget@sintef.no>

# set language, format and stuff
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# OBS: using -y is conveniently to automatically answer yes to all the questions
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get -y install sudo
RUN apt-get install curl -y
RUN apt-get install nano -y
RUN apt-get update && apt-get install -y git
RUN apt-get install libblas-dev -y && apt-get install liblapack-dev -y
RUN apt-get install gfortran -y
RUN apt-get install libpng-dev -y
RUN apt-get install python3-dev -y

# create default user account with sudo permissions
RUN useradd -m ubuntu && echo "ubuntu:ubuntu" | chpasswd && adduser ubuntu sudo
ENV PYTHON_DIR /usr/bin/python3
RUN chown ubuntu $PYTHON_DIR -R
USER ubuntu

# To expose the executable
ENV PATH="${PATH}:/home/ubuntu/.local/bin"

# downloading source code and setting up the environment
WORKDIR "/home/ubuntu"
RUN git clone https://github.com/dbouget/raidionics_rads_lib.git

# Python packages
WORKDIR "/home/ubuntu/raidionics_rads_lib"
RUN pip3 install --upgrade pip
RUN pip3 install -e .
RUN pip3 install onnxruntime-gpu==1.12.1

WORKDIR "/home/ubuntu"
USER root
# setting up a resources folder which should mirror a user folder, to "send" data/models in and "collect" the results
RUN mkdir /home/ubuntu/resources
RUN chown -R ubuntu:ubuntu /home/ubuntu/resources
RUN chown -R ubuntu:ubuntu /home/ubuntu/raidionics_rads_lib
RUN chmod -R 777 /home/ubuntu/raidionics_rads_lib
USER ubuntu
EXPOSE 8888

# CMD ["/bin/bash"]
ENTRYPOINT ["python3","/home/ubuntu/raidionics_rads_lib/main.py"]




