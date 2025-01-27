# evalVLM

This repository contains code for evaluating and benchmarking Vision Language Models (VLMs) on tabular data. Specifically, it focuses on the task of answering questions based on information presented in one or more tables.

## Repository Structure

The repository is organized as follows:

-   `results`: Directory to store the output JSONL files containing model response results and metrics.
-   `evals.ipynb`: Experimented Code.
-   `gemini_eval.py`: Python script for evaluating the Gemini model.
-   `qwen_eval.py`: Python script for evaluating the Qwen model.
-   `llava_eval.py`: Python script for evaluating the LLaVA model.
-   `phi3_eval.py`: Python script for evaluating the Phi-3 model.
-   `evaluate_metrics.py`: Python script for calculating evaluation metrics (Exact Match and F1 Score).
-   `requirements.txt`: Lists the required Python packages for running the code.
-   `utils.py`: Utility functions used by the evaluation scripts.

## Models Evaluated

The repository currently includes evaluation scripts for the following VLMs:

-   **Gemini:** `gemini_eval.py` is used to evaluate the Gemini model (specifically `gemini-exp-1206`).
-   **Qwen:** `qwen_eval.py` is used to evaluate the Qwen model (specifically `Qwen/Qwen2-VL-2B-Instruct`).
-   **LLaVA:** `llava_eval.py` is used to evaluate the LLaVA model (e.g., `llava-hf/llava-onevision-qwen2-7b-ov-hf`).
-   **Phi-3:** `phi3_eval.py` is used to evaluate the Phi-3 model (e.g., `microsoft/Phi-3.5-vision-instruct`).

## Usage

### Prerequisites

1. **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Set up API keys:**
    -   For Gemini, you need to configure your Google API key. Instructions can be found in the `gemini_eval.py` file.

### Running the Evaluation

#### Gemini

To evaluate the Gemini model, run the following command:

```bash
python gemini_eval.py --input-dir <path_to_data_directory> --output-file <path_to_output_file> --model-name gemini-exp-1206
```

-   `--input-dir`: Path to the directory containing the JSONL data file and the `table_images` subdirectory.
-   `--output-file`: Path to the output JSONL file where the results will be saved (e.g., `results/gemini_two_table_eval.jsonl`).
-   `--model-name`: The name of the Gemini model to use (default: `gemini-exp-1206`).

#### Qwen

To evaluate the Qwen model, run the following command:

```bash
python qwen_eval.py --input-dir <path_to_data_directory> --output-file <path_to_output_file> --model-path <path_to_model> --device <device> --quantization False
```

-   `--input-dir`: Path to the directory containing the JSONL data file and the `table_images` subdirectory.
-   `--output-file`: Path to the output JSONL file where the results will be saved (e.g., `results/qwen_two_table_eval.jsonl`).
-   `--model-path`: Path to the Qwen model (default: `Qwen/Qwen2-VL-2B-Instruct`).
-   `--device`: The device to run the model on (e.g., `cuda:0`, `cpu`). Default is `cuda:3`.
-   `--quantization`: Qunatization option.
-   `--quant_bit`: either 4 or 8 bit qunatization

#### LLaVA

To evaluate the LLaVA model, run the following command:

```bash
python llava_eval.py --input-dir <path_to_data_directory> --output-file <path_to_output_file> --model-path <path_to_llava_model> --device <device>
```

-   `--input-dir`: Path to the directory containing the JSONL data file and the `table_images` subdirectory.
-   `--output-file`: Path to the output JSONL file where the results will be saved (e.g., `results/llava_two_table_eval.jsonl`).
-   `--model-path`: Path to the LLaVA model (e.g., `llava-hf/llava-onevision-qwen2-7b-ov-hf`).
-   `--device`: The device to run the model on (e.g., `cuda:0`, `cpu`). Default is `cuda:0`.

#### Phi-3

To evaluate the Phi-3 model, run the following command:

```bash
python phi3_eval.py --input-dir <path_to_data_directory> --output-file <path_to_output_file> --model-path <path_to_phi3_model> --device <device>
```

-   `--input-dir`: Path to the directory containing the JSONL data file and the `table_images` subdirectory.
-   `--output-file`: Path to the output JSONL file where the results will be saved (e.g., `results/phi3_two_table_eval.jsonl`).
-   `--model-path`: Path to the Phi-3 model (e.g., `microsoft/Phi-3.5-vision-instruct`).
-   `--device`: The device to run the model on (e.g., `cuda:0`, `cpu`). Default is `cuda:3`.

#### Calculating Metrics
After generating model responses, you can calculate evaluation metrics using `evaluate_metrics.py`.

```bash
python evaluate_metrics.py --input-file <path_to_model_output_file> --output-file <path_to_metrics_output_file> --wandb-project <project_name> --wandb-entity <entity> --wandb-run-name <run_name>
```

-   `--input-file`: Path to the JSONL file containing model responses (output from `*_eval.py` scripts).
-   `--output-file`: Path to save the JSON metrics file (e.g., `results/metrics/qwen_two_table_metrics.json`).
-   `--wandb-project`: (Optional) Your Weights & Biases project name for logging metrics.
-   `--wandb-entity`: (Optional) Your Weights & Biases entity name.
-   `--wandb-run-name`: (Optional) A name for your Weights & Biases run.

### Data Format

The input data should be in a directory containing:

-   A JSONL file where each line represents a data sample.
-   A subdirectory named `table_images` containing the images of the tables referenced in the JSONL file.

Each data sample in the JSONL file should have the following format:

```json
{
    "question": "Which department has more than 1 head at a time? List the id, name and the number of heads.",
    "answer": {"columns": ["Department_ID", "Name", "count(*)"], "index": [0], "data": [[2, "Treasury", 2]]},
    "table_names": ["department", "management"],
    "table_image_ids": ["TableImg_11gu6_15.png", "TableImg_Y3m5c_5.png"],
    "original_data_index": 3
}
```

-   `question`: The question to be answered based on the tables.
-   `answer`: The ground truth answer to the question.
-   `table_image_ids`: A list of image file names (from the `table_images` directory) corresponding to the tables needed to answer the question.

### Output Format

The output file will be a JSONL file where each line represents the evaluation result for a single data sample. The format is as follows:

```json
{
    "question": "What is the name and country of origin of the artist who released a song that has \"love\" in its title?",
    "golden_answer": {"columns": ["artist_name", "country"], "index": [0], "data": [["Enrique", "USA"]]},
    "table_image_ids": ["TableImg_U9vum_6.png", "TableImg_Hn0vz_6.png"],
    "response": "[{\"artist_name\": \"Enrique\", \"country\": \"USA\", \"country_of_origin\": \"USA\"}]"
}
```

-   `question`: The original question.
-   `golden_answer`: The ground truth answer.
-   `table_image_ids`: The IDs of the tables used.
-   `response`: The model's generated response.

## System Instructions - PROMPT

The `utils.py` file contains the `SYSTEMS_INSTRUCTIONS` variable, which defines the instructions given to the models during evaluation. These instructions are designed to guide the models to:

1. Understand and reason about tabular data.
2. Carefully examine the tables and identify relevant information.
3. Formulate a clear and concise answer in natural language.
4. Avoid including SQL queries in the answer.
5. Be accurate and avoid hallucinations.
6. Provide answers in a specific JSON format.

## Notes

-   The Gemini evaluation script (`gemini_eval.py`) includes rate limiting to avoid exceeding the API usage limits.
-   The Qwen, LLaVA, and Phi-3 evaluation scripts handle CUDA out-of-memory errors by skipping the problematic data sample and freeing up memory.
-   `evals.ipynb` notebook contains experiment with the evaluation code and explore the results.
-   Make sure to adjust the `--device` argument in evaluation scripts based on your available hardware resources.
- The `evaluate_metrics.py` script calculates Exact Match and F1-score and can optionally log results to Weights & Biases.