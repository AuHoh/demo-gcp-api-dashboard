FROM python:3.8-slim

RUN apt-get update -y

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

EXPOSE 8080
ENTRYPOINT ["streamlit", "run", "main.py", "–server.port=8080", "–server.address=0.0.0.0"]