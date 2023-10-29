import json
import logging
from inference_onnx import ColaONNXPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# inferencing_instance = ColaONNXPredictor("./models/model.onnx")

def lambda_handler(event, context):
    """
    Lambda function handler for predicting linguistic acceptability of a given sentence
    """
    logger = logging.getLogger()
    logger.info("Lambda function started")

    try:
        logger.info("Processing step 1")
        if "resource" in event.keys():
            body = event["body"]
            body = json.loads(body)
            print(f"Go the input: {body['sentence']}")
    #         response = inferencing_instance.predict(body["sentence"])
    #         return {
    #             "statusCode": 200,
    #             "headers": {},
    #             "body": json.dumps(response)
    #         }
    #     else:
    #         return inferencing_instance.predict(event["sentence"])
    #     logger.info("Lambda function completed successfully")
    # except Exception as e:
    #     logger.error(f"An error occurred: {str(e)}")
    #     raise