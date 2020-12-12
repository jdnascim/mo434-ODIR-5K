FROM tensorflow/tensorflow:2.4.0rc3-gpu

USER root

RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y curl unzip vim

USER $NB_UID

WORKDIR /opt

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

WORKDIR /
