FROM python:3.12

WORKDIR /detector

# RUN mkdir /detector/models

# Add source code, configs, and data
ADD datasets ./datasets
ADD models ./models
ADD src ./src
ADD config.yaml ./config.yaml

# Add root files
ADD requirements.txt .
ADD main.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "/detector/main.py", "--config_file", "/detector/config.yaml"]
