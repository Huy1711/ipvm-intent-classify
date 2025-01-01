import boto3
import json
import time

def invoke_sagemaker_endpoint(endpoint_name, input_data):
    runtime = boto3.client('sagemaker-runtime')
    payload = json.dumps(input_data)
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=payload
    )
    result = json.loads(response['Body'].read().decode())
    return result


def test_random_prediction(endpoint_name='intent-model-endpoint'):
    sample_input = {"text": "tell me about Evolv"}
    start = time.time()
    predictions = invoke_sagemaker_endpoint(endpoint_name, sample_input)
    print(f"Prediction took: {time.time() - start:.4f} seconds")
    print(predictions)


if __name__ == "__main__":
    test_random_prediction()