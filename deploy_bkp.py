# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os

from absl import app, flags
from dotenv import load_dotenv

from agent import root_agent
from google.cloud import aiplatform

from google.adk.sessions import VertexAiSessionService

import vertexai
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp


FLAGS = flags.FLAGS
flags.DEFINE_string("project_id", None, "GCP project ID.")
flags.DEFINE_string("location", None, "GCP location.")
flags.DEFINE_string("bucket", None, "GCP bucket.")

flags.DEFINE_string(
    "initial_states_path",
    None,
    "Relative path to the initial state file, .e.g eval/itinerary_empty_default.json",
)
flags.DEFINE_string("map_key", None, "API Key for Google Places API")

flags.DEFINE_string("resource_id", None, "ReasoningEngine resource ID.")
flags.DEFINE_bool("create", False, "Creates a new deployment.")
flags.DEFINE_bool("quicktest", False, "Try a new deployment with one turn.")
flags.DEFINE_bool("delete", False, "Deletes an existing deployment.")
flags.mark_bool_flags_as_mutual_exclusive(["create", "delete", "quicktest"])


def create(env_vars: dict[str, str]) -> None:
    """Creates a new deployment."""
    print(env_vars)

    app = AdkApp(
        agent=root_agent,
        enable_tracing=True,
        env_vars=env_vars,
        # session_service_builder=session_service_builder,
    )

    remote_agent = agent_engines.create(
        app,
        display_name="Multi-Agent-ADK-new",
        description="AgentEngine Deployment",
        requirements=[
            "google-cloud-aiplatform[agent_engines,adk]",
        ],
        extra_packages=[ 
               "agent.py",       # The main package
        ],
    )
    print(f"Created remote agent: {remote_agent.resource_name}")


def delete(resource_id: str) -> None:
    remote_agent = agent_engines.get(resource_id)
    remote_agent.delete(force=True)
    print(f"Deleted remote agent: {resource_id}")


def send_message(
    session_service: VertexAiSessionService, resource_id: str, message: str
) -> None:
    """Send a message to the deployed agent."""

    session = asyncio.run(
        session_service.create_session(app_name=resource_id, user_id="user4321")
    )

    remote_agent = agent_engines.get(resource_id)

    print(f"Trying remote agent: {resource_id}")
    for event in remote_agent.stream_query(
        user_id="user4321",
        session_id=session.id,
        message=message,
    ):
        print(event)
    print("Done.")


def main(argv: list[str]) -> None:

    load_dotenv()
    env_vars = {}

    project_id = (
        FLAGS.project_id if FLAGS.project_id else os.getenv("GOOGLE_CLOUD_PROJECT")
    )
    location = FLAGS.location if FLAGS.location else os.getenv("GOOGLE_CLOUD_LOCATION")
    bucket = FLAGS.bucket if FLAGS.bucket else os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET")
    # Variables for Travel Concierge from .env
    initial_states_path = FLAGS.initial_states_path

    print(f"PROJECT: {project_id}")
    print(f"LOCATION: {location}")
    print(f"BUCKET: {bucket}")
    print(f"INITIAL_STATE: {initial_states_path}")

    env_vars = {
        "GOOGLE_GENAI_USE_VERTEXAI": os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "TRUE"),
        "GOOGLE_CLOUD_PROJECT": project_id,
        "GOOGLE_CLOUD_LOCATION": location,
        "GOOGLE_CLOUD_STORAGE_BUCKET": bucket 
    }

    if not project_id:
        print("Missing required environment variable: GOOGLE_CLOUD_PROJECT")
        return
    elif not location:
        print("Missing required environment variable: GOOGLE_CLOUD_LOCATION")
        return
    elif not bucket:
        print("Missing required environment variable: GOOGLE_CLOUD_STORAGE_BUCKET")
        return

    vertexai.init(
        project=project_id,
        location=location,
        staging_bucket=f"gs://{bucket}",
    )

    if FLAGS.create:
        create(env_vars)
    elif FLAGS.delete:
        if not FLAGS.resource_id:
            print("resource_id is required for delete")
            return
        delete(FLAGS.resource_id)
    elif FLAGS.quicktest:
        if not FLAGS.resource_id:
            print("resource_id is required for quicktest")
            return
        session_service = VertexAiSessionService(project_id, location)
        send_message(
            session_service,
            FLAGS.resource_id,
            "Hey how are you?",
        )
    else:
        print("Unknown command")


if __name__ == "__main__":
    app.run(main)