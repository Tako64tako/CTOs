FROM python:3.7
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio
#RUN pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
RUN pip install -r requirements.txt
#cv2インストール時のエラー解決用にlibgl1-mesa-devをインストール
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev
COPY . /code/