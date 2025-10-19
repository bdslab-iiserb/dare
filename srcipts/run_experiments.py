import argparse
import json
from pathlib import Path
import logging
from tqdm import tqdm
from datetime import datetime

# Import the components we have built
from data_loader import load_dataset, QueryContext
from orchestrator import DebateOrchestrator

# --- Setup Logging ---
# We'll log to both a file and the console for comprehensive tracking.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_experiment(dataset_name: str):
    """
    Main function to run the full debate framework on a specified dataset.

    Args:
        dataset_name (str): The name of the dataset to process (e.g., 'ramdocs').
    """
    # --- 1. Setup Paths ---
    project_base_dir = Path(__file__).parent.parent
    prepared_data_dir = project_base_dir / "prepared_data"
    results_dir = project_base_dir / "results"
    results_dir.mkdir(exist_ok=True) # Ensure the results directory exists

    input_filepath = prepared_data_dir / f"{dataset_name}_test_prepared.jsonl"
    
    # Create a unique output filename with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filepath = results_dir / f"{dataset_name}_results_{timestamp}.jsonl"
    
    log_file_handler = logging.FileHandler(results_dir / f"experiment_log_{timestamp}.log")
    logging.getLogger().addHandler(log_file_handler)

    logging.info(f"--- Starting Experiment on '{dataset_name}' Dataset ---")
    logging.info(f"Input data: {input_filepath}")
    logging.info(f"Output results will be saved to: {output_filepath}")

    # --- 2. Load the Dataset ---
    try:
        dataset = load_dataset(input_filepath)
    except FileNotFoundError:
        logging.error(f"Dataset file not found. Please ensure '{input_filepath}' exists.")
        return

    # --- 3. Initialize the Orchestrator ---
    # This will also initialize all the agents.
    try:
        orchestrator = DebateOrchestrator()
    except Exception as e:
        logging.error(f"Failed to initialize the DebateOrchestrator: {e}", exc_info=True)
        return

    # --- 4. Run the Debate for Each Query and Save Results ---
    with open(output_filepath, 'w', encoding='utf-8') as f_out:
        # Using tqdm for a progress bar in the console
        for query_context in tqdm(dataset, desc=f"Processing {dataset_name}"):
            logging.info(f"--- Running debate for Query ID: {query_context.query_id} ---")
            
            try:
                # This is the core call to our framework
                debate_transcript = orchestrator.run_debate(query_context)
                
                # Prepare the final result object to be saved
                result_to_save = {
                    "query_id": query_context.query_id,
                    "question": query_context.question,
                    "gold_answers": query_context.gold_answers,
                    "wrong_answers": query_context.wrong_answers,
                    "final_answer_object": debate_transcript.get("phase_3_final_answer", {}),
                    "full_debate_transcript": debate_transcript # Save the entire conversation
                }
                
                # Write the result as a new line in the output JSONL file
                f_out.write(json.dumps(result_to_save) + '\n')
                
            except Exception as e:
                logging.error(f"An error occurred during the debate for query_id '{query_context.query_id}': {e}", exc_info=True)
                # Save an error record so we know which one failed
                error_record = {
                    "query_id": query_context.query_id,
                    "error": str(e)
                }
                f_out.write(json.dumps(error_record) + '\n')
                continue

    logging.info(f"--- Experiment on '{dataset_name}' Complete. Results saved to {output_filepath} ---")


if __name__ == "__main__":
    # --- Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run the full adversarial debate framework on a specified dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ambigdocs", "faitheval", "ramdocs"],
        help="The name of the dataset to process."
    )
    
    args = parser.parse_args()
    
    run_experiment(args.dataset)
