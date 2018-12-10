FROM tensorflow/tensorflow:1.12.0-py3
MAINTAINER chenzhen "chenzhen@u51.com"

COPY poi_server.py /app/poi_server.py
COPY tokenization.py /app/tokenization.py
COPY run_poi.py /app/run_poi.py
COPY modeling.py /app/modeling.py
COPY optimization.py /app/optimization.py
COPY chinese_model/vocab.txt /app/chinese_model/vocab.txt

WORKDIR /app

RUN pip install Flask && \
    pip install tensorflow-serving-api

EXPOSE 5000

CMD python poi_server.py

#docker build -t registry.cn-hangzhou.aliyuncs.com/51zuji/poi_server:v1 .