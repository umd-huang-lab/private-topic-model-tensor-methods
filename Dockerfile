FROM continuumio/miniconda3:latest

ADD ./requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini

COPY ./privatespectrallda /privatespectrallda
RUN cd /privatespectrallda && pip install .

COPY ./ /code

WORKDIR /code
