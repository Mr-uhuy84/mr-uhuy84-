FROM python:3.12-slim
WORKDIR /app
COPY relay.py .
EXPOSE 8080
CMD ["python3", "relay.py"]
