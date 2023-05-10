FROM --platform=linux/amd64 python:3.8-bullseye
ARG TARGETARCH

# LINUX GNU + BINUTILS
RUN apt-get -y update  && apt-get -y install software-properties-common gcc-10

# Install binwalk
RUN cd /tmp && \
    git clone https://github.com/ReFirmLabs/binwalk && \
    cd binwalk && \
    python3 setup.py install

ARG OFRAK_AI_DIR=.

COPY $OFRAK_AI_DIR /ofrak_ai
WORKDIR /ofrak_ai
RUN make install
WORKDIR /
ENTRYPOINT /bin/sh
