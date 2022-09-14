FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

WORKDIR /workspace

COPY . /workspace

RUN pip uninstall opencv-python -y

RUN pip install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
