#this should be run as a notebook
import uuid
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
)

# Create a unique name for the endpoint
Taxiprice = "Taxi-price-endpoint-" + str(uuid.uuid4())[:8]

# Expect the endpoint creation to take a few minutes

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=Taxiprice,
    description="this is an online endpoint",
    auth_mode="key",
    tags={
        "training_dataset": "credit_defaults",
        "model_type": "sklearn.RandomForestRegressor",
    },
)

endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()

print(f"Endpoint {endpoint.name} provisioning state: {endpoint.provisioning_state}")

endpoint = ml_client.online_endpoints.get(name=Taxiprice)

print(
    f'Endpoint "{endpoint.name}" with provisioning state "{endpoint.provisioning_state}" is retrieved'
)

# Let's pick the latest version of the model
latest_model_version = max(
    [int(m.version) for m in ml_client.models.list(name="RFregressor")]
)
print(f'Latest model is version "{latest_model_version}" ')

# picking the model to deploy. Here we use the latest version of our registered model
model = ml_client.models.get(name=registered_model_name, version=latest_model_version)

# Expect this deployment to take approximately 6 to 8 minutes.
# create an online deployment.
# if you run into an out of quota error, change the instance_type to a comparable VM that is available.\
# Learn more on https://azure.microsoft.com/en-us/pricing/details/machine-learning/.

blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=Taxiprice,
    model=model,
    instance_type="Standard_DS3_v2",
    instance_count=1,
)

blue_deployment = ml_client.begin_create_or_update(blue_deployment).result()
