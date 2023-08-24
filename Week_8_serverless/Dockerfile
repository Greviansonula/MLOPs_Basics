FROM public.ecr.aws/lambda/python:3.10



ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG MODEL_DIR=./models
RUN mkdir $MODEL_DIR

ENV TRANSFORMERS_CACHE=$MODEL_DIR \
    TRANSFORMERS_VERBOSITY=error

# aws credentials configurations
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

# Update and upgrade package repositories, install necessary packages
RUN yum install git -y && yum -y install gcc-c++ make curl tar
RUN apt-get update && apt-get install -y sqlite3-dev

# Download and build SQLite from source using curl
# RUN curl -LO https://www.sqlite.org/2019/sqlite-autoconf-3280000.tar.gz && \
#     tar -xzf sqlite-autoconf-3280000.tar.gz && \
#     cd sqlite-autoconf-3280000 && \
#     ./configure && make && make install

# # Recompile Python with LD_RUN_PATH environment variable
# RUN cd /opt/python3.10 && \
#     LD_RUN_PATH=/usr/local/lib ./configure && \
#     LD_RUN_PATH=/usr/local/lib make && \
#     LD_RUN_PATH=/usr/local/lib make altinstall


COPY requirements_inference.txt requirements_inference.txt
RUN pip install --upgrade pip
RUN pip install -r requirements_inference.txt --no-cache-dir

COPY  ./ ./
ENV PYTHONPATH "${PYTHONPATH}:./"
ENV LC_ALL=C.UTF-8
ENV LANCG=C.UTF-8

RUN pip install 'dvc[s3]'

# # Configuring remote server in dvc
RUN dvc init --no-scm -f
RUN dvc remote add -d storage s3://basicmlops/dvcstore

# # pulling the trained model
RUN dvc pull dvcfiles/trained_model.dvc

RUN ls 
RUN python lambda_handler.py
RUN chmod -R 0755 $MODEL_DIR

CMD ["lambda_handler.lambda_handler"]
