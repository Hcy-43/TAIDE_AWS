# Rag-Prototype

This is a demonstration proof-of-concept for the sinica-medLLM project

## API KEY(s)

You should have a `.env` file with the following fields:

```
PINECONE_API_KEY=<YOUR_API_KEY>
OPENAI_API_KEY=<YOUR_API_KEY>
TAIDE_EMAIL=<YOUR_TAIDE_EMAIL>
TAIDE_PASSWORD=<YOUR_TAIDE_PASSWORD>
```

## Setup

To store the embedding data to Pinecone database, you must create the following index(es) with the following name and dimension:

| name                                        | dimension | 
|---------------------------------------------|-----------|
| sinica-rag-test-0730-text-embedding-3-large | 3072      |
| sinica-rag-test-0730-text-embedding-3-small | 1536      |
| sinica-rag-test-0730-multilingual-e5-large  | 1024      |

## Data

The file `taiwan.txt` is generated from this [wiki page](https://zh.wikipedia.org/wiki/%E8%87%BA%E7%81%A3) with [WikiText](https://wikitext.eluni.co/)

## Run the app

To host the streamlit chat app locally, run `streamlit run ./streamlit_app.py `

## Example

You could ask the question `請跟我介紹台灣的歷史` and observe the different answers.

## Docker Development

### Install nvidia-cuda-toolkit

Install `nvidia-cuda-toolkit`, follow the instructions from [nvidia](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

> Note: we use cuda==12.4 in this project

### Install nvidia-container-toolkit

To install `nvidia-container-toolkit`, follow the instructions from [nvidia](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Verify installation

To verify your installation, run the command `docker run -it --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`

### Start Docker development

Run the command `make` to build and run the docker container. Then access streamlit application through `http://0.0.0.0:8501`.
