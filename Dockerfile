FROM sia/dev:gpu-cuda9-torch1.2-torchvision0.4

RUN pip3 install transformers
RUN pip3 install tensorboard
RUN pip3 install tensorflow==1.14.0
RUN pip3 install pymongo
RUN pip3 install sacred
RUN HOROVOD_GPU_OPERATIONS=NCCL pip3 install horovod

WORKDIR /app
