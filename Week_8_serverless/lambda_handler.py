import json
from inference_onnx import ColaONNXPredictor

inferencing_instance = ColaONNXPredictor("./models/model.onnx")

def lambda_handler(evenv, context):
    """
    Lambda function handler for predicting linguistic acceptability of a given sentence
    """

    if "resource" in event.keys():
        body = event["body"]
        body = json.loads(body)
        print(f"Go the input: {body["sentence"]}")
        response = inferencing_instance.predict(body["sentence"])
        return {
            "statusCode": 200,
            "headers": {},
            "body": json.dumps(response)
        }
    else:
        return inferencing_instance.predict(event["sentence"])