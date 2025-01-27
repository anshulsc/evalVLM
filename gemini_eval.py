"""
python gemini_eval.py --input-dir ../DataVLM/mmqa_extended/two_tables --output-file results/gemini_two_table_eval.jsonl --model-name gemini-exp-1206
"""


import os
import tqdm
import google.generativeai as genai
import json
import argparse
import time
import typing_extensions as typing
from utils import find_jsonl_file, benchmark_data, SYSTEMS_INSTRUCTIONS
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def upload_to_gemini(path, mime_type=None):
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file


class Response(typing.TypedDict):
    data: list[list[str]]

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
    "response_schema": Response
}


def evaluate_gemini(data, output_file, model_name, images_dir):
    
    
    # Rate limiting variables
    rate_limit = 5  # Requests per minute
    time_period = 60  # Seconds in a minute
    requests_made = 0
    start_time = time.time()
  
    with open(output_file, "a+", encoding="utf-8") as out_file:
        
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            system_instruction=SYSTEMS_INSTRUCTIONS
            )
        for i, row in enumerate(tqdm.tqdm(data)):
            
            elapsed_time = time.time() - start_time
            if requests_made >= rate_limit and elapsed_time < time_period:
                sleep_time = time_period - elapsed_time + 60
                print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
                requests_made = 0
                start_time = time.time()
            
            if i< 272:
                continue
            
            table_length = [int(img.split("_")[-1].split(".")[0]) for img in row.get("table_image_ids", [])]
            if sum(table_length) > 1000:
                result = {
                "question": row.get("question"),
                "golden_answer": row.get("answer"),
                "table_image_ids": row.get("table_image_ids"),
                "response": "The question cannot be answered based on the provided data.",
            }
                out_file.write(json.dumps(result) + "\n")
                requests_made += 1
                
                continue
            
            files = [
                upload_to_gemini(os.path.join(images_dir, img), mime_type="image/png")
                for img in row.get("table_image_ids", [])
            ]
            prompt = f"Question: {row.get('question','')} \n answer:"
            chat_session = model.start_chat(history=[{"role": "user", "parts": files}])
            response = chat_session.send_message(prompt)
            requests_made += 1
            result = {
                "question": row.get("question"),
                "golden_answer": row.get("answer"),
                "table_image_ids": row.get("table_image_ids"),
                "response": response.text,
            }
            out_file.write(json.dumps(result) + "\n")
            if i == 500:
                break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing JSONL and table_images.")
    parser.add_argument("--output-file", default="evals/results/my_results.jsonl")
    parser.add_argument("--model-name", default="gemini-exp-1206")
    args = parser.parse_args()

    jsonl_path = find_jsonl_file(args.input_dir)
    if not jsonl_path:
        raise FileNotFoundError("No .jsonl file found in the specified directory.")
    data = benchmark_data(jsonl_path)

    images_dir = os.path.join(args.input_dir, "table_images")
    
    print(f" Jsonl file: {jsonl_path}")
    print(f" Images directory: {images_dir}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"No directory named 'table_images' found in {args.input_dir}.")

    evaluate_gemini(data, args.output_file, args.model_name, images_dir)

if __name__ == "__main__":
    main()


""""
python gemini_eval.py --input-dir ../DataVLM/mmqa_extended/two_tables --output-file results/gemini_two_table_eval.jsonl --model-name gemini-exp-1206
"""