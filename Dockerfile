FROM python:3.12-slim

# Install dependencies including OpenCV and system libraries
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

WORKDIR /app

COPY . .


RUN pip install -r requirements.txt

EXPOSE 5017

CMD ["python", "app.py"]
