
FROM python:3.8

COPY . /app
WORKDIR /app
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]