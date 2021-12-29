FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

MAINTAINER shaswot <shaswot@acsl.ics.keio.ac.jp>

# install system-wide deps for python and node
# The yqq flag is used to suppress output and assumes "Yes" to all prompts.

RUN apt-get -yqq update --fix-missing

RUN DEBIAN_FRONTEND='noninteractive' apt-get -yqq install \
        screen \
        vim \
        git \
        openssh-server \
        htop \
        cmake \
        pandoc \
        wget \
	texlive-xetex \
	texlive-fonts-recommended \
	texlive-latex-recommended \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*


# set a directory for the program 
RUN mkdir -p /workspace
WORKDIR /workspace

# copy files to the container
COPY . .

# install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r ./docker/requirements.txt

# folder to share files with host machine
RUN mkdir /stash
VOLUME stash

# define the port number the container should expose
# All running containers expose 8888
# They are redirected to other ports using docker -p <machine_port>:8888
EXPOSE 8888

# Change workdir to root so that Jupyter Lab can access the /stash folder
# /stash folder stores large dumpfiles that can be accessed from the host computer
WORKDIR /

# run bash shell
# start jupyter lab once inside the container using screen
# this allows you to access the bash shell and run multiple screens for multiple notebooks / tensorboards 
RUN /bin/bash
