#!/bin/bash 

source .env

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$HUB_TOKEN" \
    -p 8080:8000 \
    --ipc=host \
    vllm/vllm-openai:v0.3.3 \
    --model $MODEL_NAME \
    --max-model-len 16384