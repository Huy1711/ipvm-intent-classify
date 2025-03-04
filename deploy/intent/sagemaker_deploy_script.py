import os
import boto3
from sagemaker.pytorch import PyTorchModel
from sagemaker.serverless import ServerlessInferenceConfig
import tarfile
import torch

import dotenv
dotenv.load_dotenv('.env')

LOCAL_MODEL_PATH = "deploy/intent/model"
BUCKET_NAME = os.getenv("AWS_BUCKET")
S3_KEY = "sagemaker/"

def push_model_to_s3(model_path, bucket_name=BUCKET_NAME):
    s3 = boto3.client('s3')
    archive_path = compress_model(model_path)
    key = os.path.join(S3_KEY, os.path.basename(archive_path))
    s3.upload_file(
        Filename=archive_path,
        Bucket=bucket_name,
        Key=key
    )
    return f"s3://{bucket_name}/{key}"


def deploy_model(
        model_path,
        serverless=False,
        role_arn=None, 
        instance_type='ml.g4dn.xlarge', 
        endpoint_name='intent-model-endpoint', 
        torch_version='2.5.1', 
        py_version='py311'
    ):
    if role_arn is None:
        role_arn = os.getenv("AWS_ROLE_ARN")

    model_uri = push_model_to_s3(model_path=model_path)

    if torch_version != torch.__version__.split("+")[0]:
        raise ValueError("Torch Version differs.")

    pytorch_model = PyTorchModel(
        model_data=model_uri,
        role=role_arn,
        framework_version=torch_version,
        py_version=py_version,
        entry_point='inference.py',
        source_dir=os.path.join(LOCAL_MODEL_PATH, 'code')
    )

    delete_endpoint_config(endpoint_name)

    if serverless:
        serverless_config = ServerlessInferenceConfig(
            memory_size_in_mb=6144, max_concurrency=3,
        )
        
        pytorch_model.deploy(
            endpoint_name=endpoint_name,
            serverless_inference_config=serverless_config
        )
    else:
        pytorch_model.deploy(
            endpoint_name=endpoint_name,
            instance_type=instance_type,
            initial_instance_count=1
        )

   


def does_endpoint_config_exist(endpoint_config_name):
    sagemaker_client = boto3.client('sagemaker')
    try:
        sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        return True
    except sagemaker_client.exceptions.ClientError:
        return False


def delete_endpoint_config(endpoint_config_name):
    sagemaker_client = boto3.client('sagemaker')
    if does_endpoint_config_exist(endpoint_config_name):
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)


def compress_model(model_path):
    dir_name = os.path.dirname(model_path)
    file_name = os.path.basename(model_path)
    archive_file_name = f"{file_name}.tar.gz"
    with tarfile.open(os.path.join(dir_name, archive_file_name), 'w:gz') as tar:
        tar.add(os.path.join(model_path, "model.pt"), arcname="model.pt")
        tar.add(os.path.join(model_path, "code"), arcname="code")
    return os.path.join(dir_name, archive_file_name)


if __name__ == '__main__':
    deploy_model(LOCAL_MODEL_PATH, serverless=False)