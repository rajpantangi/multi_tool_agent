import os
import sys

import pytest
import vertexai


@pytest.fixture(scope="session", autouse=True)
def setup_environment():
    """Sets up the environment for all tests in the session."""

    # In a CI/CD environment, configuration should come from environment
    # variables, not hardcoded strings.
   
    #project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    #location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    #bucket_name = os.environ.get("GOOGLE_CLOUD_STORAGE_BUCKET")

    #if not all([project_id, location, bucket_name]):
     #   pytest.fail("Missing required environment variables for testing.")

    # Local version
    # Add the parent directory to the Python path to allow importing the agent module
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    project_id = "gcp-basic-371423"
    location = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
    bucket_name = "adk_sample_deployment_muti_agent"  # @param {type: "string", placeholder: "[your-bucket-name]", isTemplate: true}

    # Set environment variables for ADK to use Vertex AI
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id ## Can remove this in Build
    os.environ["GOOGLE_CLOUD_LOCATION"] = location  ## Can remove this in Build

    vertexai.init(
        project=project_id,
        location=location,
        staging_bucket=f"gs://{bucket_name}",
    )