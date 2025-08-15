
import random
import string
# from IPython.display import HTML, Markdown, display # Commented out for terminal use
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

from vertexai.preview.evaluation import EvalTask
from vertexai.preview.evaluation.metrics import (
    PointwiseMetric,
    PointwiseMetricPromptTemplate,
    TrajectorySingleToolUse,
)

# Use the environment variable if the user doesn't provide Project ID.
import os
import vertexai
from vertexai import agent_engines
from vertexai.preview import reasoning_engines
from google.cloud.aiplatform import initializer


# Read configuration from environment variables for CI/CD compatibility
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
BUCKET_NAME = os.environ.get("GOOGLE_CLOUD_STORAGE_BUCKET")

if not all([PROJECT_ID, LOCATION, BUCKET_NAME]):
    raise ValueError(
        "Missing required environment variables: "
        "GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_CLOUD_STORAGE_BUCKET"
    )

BUCKET_URI = f"gs://{BUCKET_NAME}"

EXPERIMENT_NAME = "evaluate-re-agent"  # @param {type:"string"}

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=BUCKET_URI,
)

## Helper functions

def get_id(length: int = 8) -> str:
    """Generate a uuid of a specified length (default=8)."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def display_dataframe_rows(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    num_rows: int = 3,
    display_drilldown: bool = False,    
    ) -> None:
    """Displays a subset of rows from a DataFrame, optionally including a drill-down view."""

    if columns:
        df = df[columns]

    # The following lines are for Jupyter environment.
    # base_style = "font-family: monospace; font-size: 14px; white-space: pre-wrap; width: auto; overflow-x: auto;"
    # header_style = base_style + "font-weight: bold;"

    for _, row in df.head(num_rows).iterrows():
        for column in df.columns:
            # For Jupyter environment:
            # display(
            #     HTML(
            #         f"{column.replace('_', ' ').title()}: "
            #     )
            # )
            # display(HTML(f"{row[column]}"))
            # For terminal environment:
            print(f"{column.replace('_', ' ').title()}: ")
            print(f"{row[column]}")

        # For Jupyter environment:
        # display(HTML(""))
        # For terminal environment:
        print("")

        if (display_drilldown
            and "predicted_trajectory" in df.columns
            and "reference_trajectory" in df.columns):
            # The display_drilldown function is not defined in this script.
            pass

def display_eval_report(eval_result: pd.DataFrame) -> None:
    """Display the evaluation results."""
    summary_metrics_df = pd.DataFrame.from_dict(eval_result.summary_metrics, orient="index").T
    # For Jupyter environment:
    # display(Markdown("### Summary Metrics"))
    # display(metrics_df)
    # For terminal environment:
    print("### Summary Metrics")
    print(summary_metrics_df)

    # For Jupyter environment:
    # display(Markdown(f"### Row-wise Metrics"))
    # display(eval_result.metrics_table)
    # For terminal environment:
    print(f"### Row-wise Metrics")
    #row_wise_metrics_df = pd.DataFrame.from_dict(eval_result.metrics_table, orient="index").T
    print(eval_result.metrics_table.to_string())


eval_data = {
    "prompt": [
        "Get weather in new york",
        "Get weather and time in new york",
        "Get current time in new york",
        "Get current time and weather in  new york",
    ],
    "reference_trajectory": [
        [
            {
                "tool_name": "get_weather",
                "tool_input": {"city": "new york"},
            }
        ],
        [
            {
                "tool_name": "get_weather",
                "tool_input": {"city": "new york"},
            },
            {
                "tool_name": "get_current_time",
                "tool_input": {"city": "new york"},
            },
        ],
         [
            {
                "tool_name": "get_current_time",
                "tool_input": {"city": "new york"},
            }
        ],
        [
            {
                "tool_name": "get_current_time",
                "tool_input": {"city": "new york"},
            },
            {
                "tool_name": "get_weather",
                "tool_input": {"city": "new york"},
            }
        ],
       
    ],
}

eval_data1 = {
    "prompt": [
        "Get weather and time in new york"
    ],
    "reference_trajectory": [
        [
            {
                "tool_name": "get_weather",
                "tool_input": {"city": "new york"},
            },
            {
                "tool_name": "get_current_time",
                "tool_input": {"city": "new york"},
            },
        ],
    ],
}

eval_sample_dataset = pd.DataFrame(eval_data)

#display_dataframe_rows(eval_sample_dataset, num_rows=3)

## Calling the deployed agent

RESOURCE_ID = os.environ.get("REASONING_ENGINE_ID")
if not RESOURCE_ID:
    raise ValueError("REASONING_ENGINE_ID environment variable not set.")

remote_agent_for_query = agent_engines.get(RESOURCE_ID)

#print (remote_agent_for_query)

'''
remote_session = remote_agent_for_query.create_session(user_id="u_456")
response_text = []
for event in remote_agent_for_query.stream_query(
    message="Get me the time in new york",
    session_id=remote_session["id"],
    user_id="u_456",
):
    for part in event["content"]["parts"]:
        if "text" in part:
            response_text.append(part["text"])

print("".join(response_text))
'''

def run_ga_agent(prompt: str) -> dict:
    """
    Wrapper function to invoke the GA Agent Engine and format the output
    for the preview evaluation framework.
    """
    final_text_output = []
    predicted_trajectory = []

    try:
        user_id = get_id()
        session = remote_agent_for_query.create_session(user_id=user_id)
        for event in remote_agent_for_query.stream_query(
            message=prompt,
            session_id=session["id"],
            user_id=user_id,
        ):
            content = event.get("content", {})
            # We only care about what the model outputs.
            # The model's output can contain text and/or function calls.
            if content.get("role") == "model":
                for part in content.get("parts", []):
                    if "function_call" in part:
                        # This is a tool call from the model
                        function_call = part["function_call"]
                        tool_name = function_call.get("name")
                        tool_input = function_call.get("args")
                        if tool_name:
                            predicted_trajectory.append(
                                {
                                    "tool_name": tool_name,
                                    "tool_input": tool_input,
                                }
                            )
                    elif "text" in part:
                        # This is a text response from the model
                        final_text_output.append(part["text"])

        response_dict= {
            "response": "".join(final_text_output),
            "predicted_trajectory": predicted_trajectory,
        }
        print("Final response created: \n", response_dict)
        return response_dict

    except Exception as e:
        print(f"Error during agent execution: {e}")
        return {"response": f"Error: {e}", "predicted_trajectory": []}


# Agent Eval

trajectory_metrics = [
    "trajectory_exact_match",
    "trajectory_precision",
    "trajectory_recall",
]

single_tool_usage_metrics = [TrajectorySingleToolUse(tool_name="get_weather")]

EXPERIMENT_RUN_SINGLE_TOOL_METRICS = f"single-metric-eval-{get_id()}"
EXPERIMENT_RUN_TRAJECTORY_METRICS = f"trajectory-metric-eval-{get_id()}"

trajectory_eval_task = EvalTask(
    dataset=eval_sample_dataset,
    metrics=trajectory_metrics,
    experiment=EXPERIMENT_NAME,
)

single_tool_eval_task = EvalTask(
    dataset=eval_sample_dataset,
    metrics=single_tool_usage_metrics,
    experiment=EXPERIMENT_RUN_SINGLE_TOOL_METRICS,
)

eval_result = trajectory_eval_task.evaluate(
    runnable=run_ga_agent, experiment_run_name=EXPERIMENT_RUN_SINGLE_TOOL_METRICS
)

display_eval_report(eval_result)
