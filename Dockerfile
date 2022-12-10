FROM python:3.8.0-slim

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

EXPOSE 5000
COPY . .

ENTRYPOINT ["bash", "launch.sh"]