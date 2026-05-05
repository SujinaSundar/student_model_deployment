import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

role = "arn:aws:iam::479368492496:role/service-role/AmazonSageMakerAdminIAMExecutionRole"

boto_session = boto3.Session(region_name="ap-south-1")
session = sagemaker.Session(boto_session=boto_session)

import time

endpoint_name = f"student-pass-endpoint-{int(time.time())}"

model = SKLearnModel(
    model_data="s3://cicd-student/model.tar.gz",
    role=role,
    entry_point="inference.py",  
    framework_version="1.2-1",
    sagemaker_session=session
)

# 🔍 Check if endpoint exists
sm_client = boto3.client("sagemaker", region_name="ap-south-1")

try:
    sm_client.describe_endpoint(EndpointName=endpoint_name)
    endpoint_exists = True
except:
    endpoint_exists = False

# 🚀 Deploy logic
if endpoint_exists:
    print("🔄 Updating existing endpoint...")
    predictor = model.deploy(
        instance_type="ml.t2.medium",
        initial_instance_count=1,
        endpoint_name=endpoint_name,
        update_endpoint=True
    )
else:
    print("🚀 Creating new endpoint...")
    predictor = model.deploy(
        instance_type="ml.t2.medium",
        initial_instance_count=1,
        endpoint_name=endpoint_name
    )

print("✅ Deployment complete")