FROM huggingface/transformers-pytorch-cpu:latest

COPY  ./ /app
WORKDIR /app

# Install requirements
RUN pip install "dvc[gdrive]"
RUN pip install -r requirements.txt

# Initialize dvc
RUN dvc init --no-scm
# Configuring remote dvc
RUN dvc remote add -d storage s3://basicmlops/dvcstore

# pulling the trained model
RUN dvc pull dvcfiles/trained_model.dvc

ENV LC_ALL=C.UTF-8
ENV LANCG=C.UTF-8

# running the application
EXPOSE 8000
CMD  ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]