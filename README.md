# OFRAK AI

## OFRAK AI Overview

This pacakge contains AI-powered OFRAK components, using the power of [OFRAK](https://github.com/redballoonsecurity/ofrak) to unpack, analyze, modify, and repack binaries along with the latest large language models to assist with binary analysis and reverse engineering.

## Installation

```
git clone https://github.com/redballoonsecurity/ofrak-ai.git
cd ofrak_ai
pip install .
```

## FAQ

Q. How do I use my OpenAI API key?

A. Run the command `export OPENAI_API_KEY='<your key>'`, and optionally `export OPENAI_ORGANIZATION='<your organization>'`. If you'd like this to be persistent, add it to your `.bashrc` or appropriate file.

Q. Why am I encountering an APIConnectionError even though my `aiohttp` install is up-to-date?

A. Run the `Install Certificates.command` script that comes bundled with your Python install.
