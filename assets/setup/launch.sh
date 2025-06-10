sudo docker run -it --rm --gpus all --ipc=host --shm-size 80G env202408:latest /bin/bash -c "export NCCL_SHM_DISABLE=1 && export NCCL_P2P_DISABLE=1 && export NCCL_ASYNC_ERROR_HANDLING=1 && exec bash"
