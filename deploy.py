import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# ✅ FIXED: use role ARN (not get_execution_role)
role = "arn:aws:iam::479368492496:role/service-role/AmazonSageMakerAdminIAMExecutionRole"

boto_session = boto3.Session(region_name="ap-south-1")
session = sagemaker.Session(boto_session=boto_session)

model = SKLearnModel(
    model_data="s3://cicd-student/model.tar.gz",
    role=role,
    framework_version="1.2-1",
    sagemaker_session=session
)

predictor = model.deploy(
    instance_type="ml.t2.medium",
    initial_instance_count=1,
    endpoint_name="student-pass-endpoint",
    update_endpoint=True
)

print("🚀 Deployment started")