FROM debian:stretch

RUN apt-get update 
RUN apt-get install -y build-essential cmake pkg-config libssl-dev libffi-dev python3-setuptools 
RUN apt-get install -y libjpeg-dev libtiff5-dev libpng-dev
RUN apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev 
RUN apt-get install -y libxvidcore-dev libx264-dev 
RUN apt-get install -y libgtk2.0-dev libgtk-3-dev 
RUN apt-get install -y libatlas-base-dev gfortran 
RUN apt-get install -y python3-dev 
RUN apt-get install -y python3-pip
RUN apt-get install -y curl zip

RUN python3 -m pip install --upgrade pip

RUN apt-get install -y libqtgui4
#RUN modprobe bcm2835-v4l2
RUN apt-get install -y libqt4-test

RUN apt-get install -y libilmbase-dev
RUN apt-get install -y libopenexr-dev
RUN apt-get install -y libgstreamer1.0-dev

RUN apt-get install -y git

COPY requirements.txt /src/requirements.txt

RUN pip3 install -r /src/requirements.txt

EXPOSE 5000

RUN mkdir -p /model
RUN mkdir -p /api
COPY api/ /api/





