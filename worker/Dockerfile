FROM danieltromp/py36-cv2:latest
MAINTAINER "Daniël Tromp" <drpgmtromp@gmail.com>

ENV TERM=xterm \
    TZ='Europe/Amsterdam' \
    DEBIAN_FRONTEND=noninteractive

RUN echo $TZ > /etc/timezone

RUN apt-get update  --fix-missing \
    && apt-get -y upgrade \
    && apt-get --yes --no-install-recommends install apt-utils tzdata \
                                locales tzdata ca-certificates sudo \
    && dpkg-reconfigure -f noninteractive tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install sklearn Keras==2.2.4 matplotlib tensorflow==1.13.1 \
                    celery minio redis

RUN git clone https://github.com/DanielTromp/Demo_App_AI-MicroServices.git
RUN cp -ar /Demo_App_AI-MicroServices/worker/ /app/

WORKDIR /app/
RUN mkdir -p tmp
RUN wget -q https://www.dropbox.com/s/3pz96kng6hcupf5/age_model_weights.h5 -P models/
RUN wget -q https://www.dropbox.com/s/laum0pct5exj73r/gender_model_weights.h5 -P models/

CMD ["celery", "-A", "worker.celery", "worker", "--loglevel=info"]

# docker build --no-cache  --tag worker ./worker/
# docker run -it worker