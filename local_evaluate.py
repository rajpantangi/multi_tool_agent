import asyncio
import os
import sys

import vertexai

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from google.adk.evaluation import AgentEvaluator


PROJECT_ID = "gcp-basic-371423"  # @param {type: "string", placeholder: "[your-project-id]", isTemplate: true}

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

BUCKET_NAME = "adk_sample_deployment_muti_agent"  # @param {type: "string", placeholder: "[your-bucket-name]", isTemplate: true}

BUCKET_URI = f"gs://{BUCKET_NAME}"

EXPERIMENT_NAME = "evaluate-re-agent"  # @param {type:"string"}

# Set environment variables for ADK to use Vertex AI.
# This is necessary for the local AgentEvaluator to connect to the backend.
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=BUCKET_URI,
)

async def main():
    """Runs a local evaluation of the agent."""

    # The agent_module is the name of the folder containing your agent.py
    # The ADK will look for a `root_agent` object inside `agent.py`.
    agent_module = "multi_tool_agent"
    eval_dataset_path = "eval/multi_tool_new.test.json"

    print(f"Starting local evaluation for agent: {agent_module}")
    print(f"Using evaluation dataset: {eval_dataset_path}")

    # The AgentEvaluator will automatically find `eval/test_config.json`
    # relative to the test file path.
    try:
        await AgentEvaluator.evaluate(
            agent_module=agent_module,
            eval_dataset_file_path_or_dir=eval_dataset_path,
        )
        print("\n✅ Local evaluation completed successfully and all criteria passed.")
    except AssertionError as e:
        print(f"\n❌ Local evaluation failed: {e}")
        # Exit with a non-zero code to indicate failure, useful for CI/CD
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())