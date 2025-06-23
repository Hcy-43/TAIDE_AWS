.PHONY: all build dev stop
all: build dev
build:
	docker build -t pytorch-gpu . -f Dockerfile
dev:
	docker run --name pytorch-container --gpus all -it --rm -v $(shell pwd):/app -p 8501:8501 pytorch-gpu 
stop:
	docker kill pytorch-container
clean:
	docker rmi pytorch-gpu

