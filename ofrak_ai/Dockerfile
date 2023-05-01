FROM --platform=linux/amd64 python:3.8-bullseye
ARG TARGETARCH

# LINUX GNU + BINUTILS
RUN apt-get -y update  && apt-get -y install software-properties-common gcc-10

# Install binwalk
RUN cd /tmp && \
    git clone https://github.com/ReFirmLabs/binwalk && \
    cd binwalk && \
    python3 setup.py install

# TODO: Cut down on the number of pre-installed dependencies
### Python dependencies from the ofrak_core requirements file[s]
RUN python3 -m pip install --upgrade pip &&\
   python3 -m pip install 'aiohttp~=3.8.1' 'aiohttp-cors~=0.7.0' 'beartype~=0.12.0' 'black==22.6.0' 'fdt==0.3.3' 'importlib-metadata>=1.4' 'intervaltree==3.1.0' 'keystone-engine==0.9.2' 'lief==0.12.3' 'orjson~=3.8.7' 'pefile==2023.2.7' 'pycdlib==1.12.0' 'python-magic; platform_system != "Windows"' 'python-magic-bin; platform_system == "Windows"' 'reedsolo==1.7.0' 'sortedcontainers==2.2.2' 'synthol~=0.1.1' 'typeguard~=2.13.3' 'ubi-reader==0.8.5' 'xattr==0.10.1; platform_system != "Windows"' 'mkdocs==1.2.3' 'mkdocs-autorefs==0.3.0' 'mkdocstrings==0.16.2' 'mkdocs-literate-nav==0.4.0' 'mkdocs-material==7.3.3' 'mkdocs_gen_files==0.3.3' 'jinja2==3.0.0' 'pytkdocs>=0.12.0' 'PyYAML>=5.4' 'autoflake==1.4' 'pytest' 'hypothesis~=6.39.3' 'hypothesis-trio' 'trio-asyncio' 'mypy==0.942' 'ofrak_angr~=1.0' 'ofrak_capstone~=1.0' 'psutil~=5.9' 'pyelftools==0.29' 'pytest-aiohttp' 'pytest-asyncio==0.19.0' 'pytest-lazy-fixture' 'pytest-cov' 'pytest-xdist' 'requests' 'fun-coverage==0.2.0'

### Python dependencies from the disassemblers/ofrak_ghidra requirements file[s]
RUN python3 -m pip install --upgrade pip &&\
   python3 -m pip install 'PyYAML>=5.4' 'aiohttp~=3.8.1'


ARG OFRAK_AI_DIR=.

COPY $OFRAK_AI_DIR /ofrak_ai
WORKDIR /ofrak_ai
RUN make install
WORKDIR /
