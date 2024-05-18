FROM python:3.10-slim

ENV PYTHONUNBUFFERED=True \
    PORT=9090

WORKDIR /app
COPY requirements.txt .
# RUN pip install git+https://github.com/rwightman/pytorch-image-models.git
# RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gitpython
RUN apt-get update && apt-get install -y git

RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . ./

CMD exec gunicorn --preload --bind :$PORT --workers 1 --threads 8 --timeout 0 _wsgi:app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y