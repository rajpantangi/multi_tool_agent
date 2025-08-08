import os
import sys

import pytest
import vertexai


@pytest.fixture(scope="session", autouse=True)
def setup_environment():
    """Sets up the environment for all tests in the session."""

    # In a CI/CD environment, configuration should come from environment
    # variables, not hardcoded strings.
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    bucket_name = os.environ.get("GOOGLE_CLOUD_STORAGE_BUCKET")

    if not all([project_id, location, bucket_name]):
        pytest.fail(
            "Missing required environment variables for testing: "
            "GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_CLOUD_STORAGE_BUCKET"
        )

    # Add the parent directory to the Python path to allow importing the agent module
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    # Set environment variables for ADK to use Vertex AI
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    os.environ["GOOGLE_CLOUD_LOCATION"] = location

    vertexai.init(
        project=project_id,
        location=location,
        staging_bucket=f"gs://{bucket_name}",
    )