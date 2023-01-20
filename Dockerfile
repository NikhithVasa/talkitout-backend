FROM python:3.7
WORKDIR /opt/app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python","./src/main.py"]
