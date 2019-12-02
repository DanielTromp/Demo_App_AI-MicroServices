FROM python:3.7-alpine3.8
MAINTAINER "Daniël Tromp" <drpgmtromp@gmail.com>

RUN apk add --no-cache --virtual .build-deps binutils gcc git libc-dev make \
    && pip install flask gunicorn==19.7.1 minio celery redis \
    && git clone https://github.com/DanielTromp/Demo_App_AI-MicroServices.git \
    && apk del .build-deps binutils gcc libc-dev make

RUN cp -ar /Demo_App_AI-MicroServices/webapp/ /app/

WORKDIR /app/
RUN mkdir -p tmp

EXPOSE 5556

CMD ["gunicorn", "--bind=0.0.0.0:5555", "app"]

# docker build --no-cache --tag webapp ./webapp/
# docker run --detach -p 5555:5555 webapp
# docker run -it -p 5555:5555 webapp /bin/bash