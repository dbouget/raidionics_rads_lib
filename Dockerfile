
# creates virtual ubuntu in docker image
FROM ubuntu:22.04

# maintainer of docker file
MAINTAINER David Bouget <david.bouget@sintef.no>

# set language, format and stuff
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# OBS: using -y is conveniently to automatically answer yes to all the questions
# installing python3 with a specific version
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt update
RUN apt install python3.7 -y
RUN apt install python3.7-distutils -y
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

# installing other libraries
RUN apt-get install python3-pip -y && \
    apt-get -y install sudo
RUN apt-get install curl -y
RUN apt-get install nano -y
RUN apt-get update && apt-get install -y git
RUN apt-get install libblas-dev -y && apt-get install liblapack-dev -y
RUN apt-get install gfortran -y
RUN apt-get install libpng-dev -y
RUN apt-get install python3-dev -y
# RUN apt-get -y install cmake curl

# create default user account with sudo permissions
RUN useradd -m ubuntu && echo "ubuntu:ubuntu" | chpasswd && adduser ubuntu sudo
ENV PYTHON_DIR /usr/bin/python3
RUN chown ubuntu $PYTHON_DIR -R
USER ubuntu

# To expose the executable
ENV PATH="${PATH}:/home/ubuntu/.local/bin"

# downloading source code (not necessary, mostly to run the test scripts)
WORKDIR "/home/ubuntu"
RUN git clone https://github.com/dbouget/raidionics-rads-lib.git --recurse-submodules

# Python packages
RUN pip3 install --upgrade pip
RUN pip3 install git+https://github.com/dbouget/raidionics-rads-lib.git
RUN pip3 install onnxruntime-gpu==1.12.1

USER root
RUN mkdir /home/ubuntu/ANTsX
WORKDIR "/home/ubuntu/ANTsX"
COPY ANTsX/ $WORKDIR
WORKDIR "/home/ubuntu"

# setting up a resources folder which should mirror a user folder, to "send" data/models in and "collect" the results
RUN mkdir /home/ubuntu/resources
RUN chown -R ubuntu:ubuntu /home/ubuntu/resources
RUN chown -R ubuntu:ubuntu /home/ubuntu/raidionics-rads-lib
RUN chmod -R 777 /home/ubuntu/raidionics-rads-lib
USER ubuntu
EXPOSE 8888

# CMD ["/bin/bash"]
ENTRYPOINT ["python3","/home/ubuntu/raidionics-rads-lib/main.py"]




