FROM dongwei2021/centos_py37_tf25_gpu:v1.0.0


# time zone set
WORKDIR /usr/share
ADD ./zoneinfo ./zoneinfo
RUN  ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN echo "Asia/Shanghai" > /etc/timezone



# 复制文件
WORKDIR /ntt
ADD ./datasets ./datasets
WORKDIR /opt
ADD ./log ./log
ADD ./utils ./utils
ADD ./config ./config
ADD .env .

ADD flasktest.py .
ADD release.sh .

ADD server.py .
ADD server.sh .

ADD train.py .
ADD train.sh .
