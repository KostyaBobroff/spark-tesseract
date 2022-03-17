ARG IMAGE_VARIANT=slim-buster
ARG OPENJDK_VERSION=8
ARG PYTHON_VERSION=3.9.8
FROM bitnami/spark:3-debian-10
# FROM python:${PYTHON_VERSION}-${IMAGE_VARIANT} AS py3
# FROM openjdk:${OPENJDK_VERSION}-${IMAGE_VARIANT}

# COPY --from=py3 / /
USER root

WORKDIR /code
RUN apt-get update \
    && apt-get install tesseract-ocr \
     tesseract-ocr-all \
     poppler-utils \
    ffmpeg \
     libsm6 \
     libxext6 \
     python3 \
     python3-pip \
     wkhtmltopdf -y

 RUN python -m pip install --upgrade pip

RUN pip install pytesseract PyPDF2 opencv-contrib-python pdf2image matplotlib notebook pdfkit uuid

ADD . .

# ARG PYSPARK_VERSION=3.2.0
# RUN pip install  pyspark==${PYSPARK_VERSION}