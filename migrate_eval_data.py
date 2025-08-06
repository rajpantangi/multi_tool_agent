import os

from google.adk.evaluation import AgentEvaluator


def main():
    """
    Migrates an evaluation dataset from the old format to the new EvalSet schema.
    """
    # Define file paths relative to this script's location
    script_dir = os.path.dirname(__file__)
    old_eval_file = os.path.join(script_dir, "eval", "multi_tool_full.test.json")
    new_eval_file = os.path.join(script_dir, "eval", "multi_tool_full_evalset.test.json")

    print(f"Starting migration for: {old_eval_file}")

    if not os.path.exists(old_eval_file):
        print(f"Error: Source file not found at '{old_eval_file}'")
        return

    try:
        AgentEvaluator.migrate_eval_data_to_new_schema(
            old_eval_data_file=old_eval_file, new_eval_data_file=new_eval_file
        )
        print(f"Successfully migrated data to: {new_eval_file}")
    except Exception as e:
        print(f"An error occurred during migration: {e}")

if __name__ == "__main__":
    main()
