FROM hub.infervision.com/vendor/ubuntu:14.04
RUN echo "deb http://mirrors.infervision.com/ubuntu/    trusty    main    multiverse    restricted    universe \n \
deb http://mirrors.infervision.com/ubuntu/    trusty-backports    main    multiverse    restricted    universe \n \
deb http://mirrors.infervision.com/ubuntu/    trusty-security    main    multiverse    restricted    universe \n \
deb http://mirrors.infervision.com/ubuntu/    trusty-updates    main    multiverse    restricted    universe" > /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common wget curl && \
    rm -rf /var/lib/apt/lists/*
ARG HTTPS_PROXY
RUN echo "Acquire::https::proxy \"${HTTPS_PROXY}\";" && \
    echo "Acquire::https::proxy \"${HTTPS_PROXY}\";" > /etc/apt/apt.conf && \
    add-apt-repository ppa:fkrull/deadsnakes-python2.7 && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 5BB92C09DB82666C && \
    apt-get update && \
    apt-get install -y python2.7 && \
     ln -s /usr/bin/python2.7 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/* && rm -rf /etc/apt/apt.conf 
RUN curl https://bootstrap.pypa.io/get-pip.py | python - && \
    echo "[global]\n\
trusted-host = repos.infervision.com\n\
index-url = https://repos.infervision.com/repository/pypi/simple\n" > /etc/pip.conf
