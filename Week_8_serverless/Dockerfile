FROM amazon/aws-lambda-python

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG MODEL_DIR=./models
RUN mkdir $MODEL_DIR

ENV TRANSFORMERS_CACHE=$MODEL_DIR \
    TRANSFORMERS_VERBOSITY=error


# aws credentials configurations
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY


RUN yum update -y && \
    yum upgrade -y && \
    yum -y install sqlite sqlite-devel gcc-c++
COPY requirements_inference.txt requirements_inference.txt
COPY  ./ ./
ENV PYTHONPATH "${PYTHONPATH}:./"
ENV LC_ALL=C.UTF-8
ENV LANCG=C.UTF-8
RUN pip install 'dvc[s3]'
# Configuring remote server in dvc
RUN dvc init --no-scm -f
RUN dvc remote add -d storage s3://basicmlops/dvcstore
# pulling the trained model
RUN dvc pull dvcfiles/trained_model.dvc
RUN ls
RUN python lambda_handler.py
RUN chmod -R 0755 $MODEL_DIR
CMD ["lambda_handler.lambda_handler"]