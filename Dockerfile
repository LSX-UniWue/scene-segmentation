FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

## config
ARG USER=zehe
ARG UID=1052
# set to 1 to install conda:
ARG INSTALL_CONDA=0

## setup (you don't have to touch this)
RUN touch `date` && apt-get update

RUN apt install python3 python3-pip python3-dev python3-venv zsh byobu htop vim git wget lsof fuse parallel rsync -y

RUN adduser ${USER} --uid ${UID} --home /home/ls6/${USER}/ --disabled-password --gecos "" --no-create-home
RUN mkdir -p /home/ls6/${USER}
RUN chown -R ${USER} /home/ls6/${USER}

RUN mkdir -p /pip
RUN chown -R ${USER} /pip


USER ${UID}
RUN python3 -m venv /pip

ADD requirements.txt .

RUN bash -c "source /pip/bin/activate && pip3 install --upgrade --force-reinstall --no-cache-dir torch==2.1.0 triton \
  --index-url https://download.pytorch.org/whl/cu118"
RUN bash -c "source /pip/bin/activate && pip3 install numpy scipy pandas sympy"
RUN bash -c "source /pip/bin/activate && pip3 install 'unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git'"
RUN bash -c "source /pip/bin/activate && pip3 install -r requirements.txt"
RUN bash -c "source /pip/bin/activate && pip3 install ntfy[matrix]"

ENV PYTHONUNBUFFERED=1
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8


USER 0

RUN if [ "${INSTALL_CONDA}" = "1" ]; then bash -c 'mkdir /conda && chown ${UID} conda && \
\
cd /conda && \
/bin/bash -c "wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O ./miniconda.sh" && \
chmod 0755 ./miniconda.sh && \
/bin/bash -c "./miniconda.sh -b -p ./conda" && \
\
ln -s "/conda/conda/bin/conda" "/usr/local/bin/conda" && \
rm ./miniconda.sh && \
\
chown -R ${UID} /conda'; \
fi

USER ${UID}