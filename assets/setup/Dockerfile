FROM nvcr.io/nvidia/pytorch:24.06-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN pip install huggingface_hub==0.25.0
RUN pip install thop==0.1.1.post2209072238
RUN pip install lightning==2.5.1
RUN pip install transformers==4.45.0
RUN pip install torchvision==0.19.0
RUN pip install neptune==1.13.0
RUN pip install torchaudio==2.4.0
RUN pip install h5py
