import sys

import pytest
from google.adk.evaluation import AgentEvaluator


@pytest.mark.asyncio
async def test_multi_tool_agent_evaluation(setup_environment):
    """
    Tests the multi_tool_agent's basic ability via a local evaluation.
    """
    # In the Cloud Build environment, the workspace root is the project directory.
    # The agent is defined in `agent.py`, so the module name is "agent".
    agent_module = "workspace"
    eval_dataset_path = "eval/multi_tool_full_evalset.test.json"

    print(f"\nStarting local evaluation for agent: {agent_module}")
    print(f"Using evaluation dataset: {eval_dataset_path}")

    try:
        await AgentEvaluator.evaluate(
            agent_module=agent_module,
            eval_dataset_file_path_or_dir=eval_dataset_path,
        )
        print("\n✅ Local evaluation completed successfully and all criteria passed.")
    except AssertionError as e:
        print(f"\n❌ Local evaluation failed: {e}")
        # Re-raise the assertion to make sure pytest marks the test as failed
        raise