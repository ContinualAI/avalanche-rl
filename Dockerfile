FROM python:3.8-slim-buster
LABEL maintainer="nicolo.lucchesi@gmail.com"

# Install the C compiler tools
RUN apt-get update -y && \
  apt-get install build-essential -y && \
  apt-get install -y wget && \
  apt-get install -y python3-opencv && \
  apt-get install -y unar && \
  pip install --upgrade pip
RUN apt-get install git -y

# add a user or pip will complain
RUN adduser avalanche-user
RUN chown avalanche-user: /
USER avalanche-user
COPY --chown=avalanche-user:avalanche-user . /home/avalanche-user/app 

WORKDIR /home/avalanche-user/app
RUN pip install --user -r requirements.txt
RUN pip install --user .