FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    python3-pip \
    python3-dev 

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch and other dependencies
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other python packages
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

# Download the model

RUN huggingface-cli download Qwen/Qwen2-0.5B-Instruct

# Set the working directory
WORKDIR /app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
