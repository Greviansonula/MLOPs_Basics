FROM amazon/aws-lambda-python:3.9

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

# Install system dependencies and tools
RUN yum update -y && yum install -y sqlite-devel gcc
RUN yum install -y libreadline zlib openssl


ARG MODEL_DIR=./models
RUN mkdir $MODEL_DIR

ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"

ENV TRANSFORMERS_CACHE=$MODEL_DIR \
    TRANSFORMERS_VERBOSITY=error

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

# RUN yum install git -y && yum -y install gcc-c++
COPY requirements_inference.txt requirements_inference.txt
RUN pip install -r requirements_inference.txt --no-cache-dir
COPY ./ ./
# Copy SQLite files into the image
COPY /architecture/sqlite_layer/sqlite/lib/libsqlite3.so /usr/local/lib
ENV PYTHONPATH "${PYTHONPATH}:./"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN pip install "dvc[s3]"
# configuring remote server in dvc
RUN dvc init --no-scm -f
RUN dvc remote add -d model-store s3://basicmlops/dvcstore

# # pulling the trained model
# RUN dvc pull dvcfiles/trained_model.dvc
# RUN dvc pull dvcfiles/*

# COPY pre-pulled data and models (replace with actual file paths)
COPY dvcfiles/trained_model.dvc dvcfiles/trained_model.dvc
COPY dvcfiles/* dvcfiles/*

RUN python lambda_handler.py
RUN chmod -R 0755 $MODEL_DIR
CMD ["lambda_handler.lambda_handler"] 