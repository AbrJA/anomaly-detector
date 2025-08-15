FROM python:3.12

RUN pip install -r requirements.txt

WORKDIR /detector

RUN mkdir /detector/models

ADD datasets /detector/datasets
ADD src /detector/src
ADD requirements.txt .
ADD main.py .

CMD ["python", "-i", "main.py"]
