FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
# use an older system (18.04) to avoid opencv incompatibility (issue#3524)

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo ninja-build
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_IfD} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

# Ensure CUDA environment variables are available for appuser
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/home/appuser/.local/bin:${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH}
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user tensorboard cmake onnx   # cmake from apt-get is too old
RUN pip install --user "Pillow<10.0.0"  # Compatible with detectron2 v0.6
RUN pip install --user torch==1.10 torchvision==0.11.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2 - use version compatible with PyTorch 1.10
# For PyTorch 1.10, we need detectron2 v0.6 or compatible commit
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo && \
    cd detectron2_repo && \
    git checkout v0.6
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
# Verify CUDA installation and PyTorch headers (torch.cuda.is_available() will be False during build as GPU is not accessible)
# This is expected and normal - CUDA extensions will still compile correctly
RUN python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; import os; print("CUDA_HOME:", CUDA_HOME); print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", "not set")); print("torch.cuda.is_available():", torch.cuda.is_available()); print("CUDA_HOME from torch:", CUDA_HOME); import torch; print("PyTorch version:", torch.__version__); print("PyTorch path:", torch.__file__)' \
    | tee /home/appuser/output.txt

# Build detectron2 from source
RUN cd detectron2_repo && pip install --user -e .

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
WORKDIR /home/appuser/detectron2_repo
RUN pip install playwright
RUN playwright install chromium
RUN playwright install-deps 

# run detectron2 under user "appuser":
# wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# python3 demo/demo.py  \
	#--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
	#--input input.jpg --output outputs/ \
	#--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
