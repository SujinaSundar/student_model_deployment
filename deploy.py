from sagemaker.sklearn.model import SKLearnModel
from sagemaker import get_execution_role
import sagemaker

role = get_execution_role()

session = sagemaker.Session(default_bucket="cicd-student")

model = SKLearnModel(
    model_data="s3://cicd-student/model.tar.gz",
    role=role,
    entry_point="inference.py",
    framework_version="1.2-1",
    sagemaker_session=session
)

predictor = model.deploy(
    instance_type="ml.t2.medium",
    initial_instance_count=1
)