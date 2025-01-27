""" python phi3_eval.py --input-dir ../DataVLM/mmqa_extended/three_tables \
    --output-file results/phi3_three_table_eval.jsonl \
    --model-path microsoft/Phi-3.5-vision-instruct \
    --device cuda:3
"""

import os
import tqdm
import json
import argparse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from utils import find_jsonl_file, benchmark_data, SYSTEMS_INSTRUCTIONS_2

def load_phi3_model(model_path, device="cuda:3"):
    bnb_config = BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_quant_type="nf4",
          bnb_4bit_use_double_quant=True,
          bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": device},
        trust_remote_code=True,
        quantization_config=bnb_config,
        _attn_implementation = 'eager'
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor

def prepare_phi3_input(data_row, images_dir, processor):
    images = []
    print(f"Length of images: {data_row['table_image_ids']}")
    for image_id in data_row["table_image_ids"]:
        image_path = os.path.join(images_dir, image_id)
        images.append(Image.open(image_path))
    
    
    print(f"Length of images: {len(images)}")
    
    # Phi-3 uses special image placeholders
    image_placeholders = "\n".join([f"<|image_{i+1}|>" for i in range(len(images))])
    prompt = f"\n Please respond in JSON format as mention above. \n\n Question: {data_row['question']}\n answer:"
    
    messages = [{
        "role": "user", 
        "content": f"{SYSTEMS_INSTRUCTIONS_2}\n\n{image_placeholders}\n\n{prompt}"
    }]
    
    text = processor.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = processor(
        text=text,
        images=images,
        return_tensors="pt"
    )
    return inputs

def generate_phi3_response(model, processor, inputs):
    with torch.inference_mode():
        inputs = inputs.to(model.device)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            eos_token_id=processor.tokenizer.eos_token_id
        )
        generated_ids_trimmed = generated_ids[:, inputs['input_ids'].shape[1]:]
        return processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

def evaluate_phi3(data, output_file, model_path, images_dir, device="cuda:3"):
    model, processor = load_phi3_model(model_path, device)
    print("Evaluation Started....")

    with open(output_file, "a+", encoding="utf-8") as out_file:
        for i, row in enumerate(tqdm.tqdm(data)):
            
            if i < 2400:
                continue
            # Existing table length check
            table_length = [int(img.split("_")[-1].split(".")[0]) for img in row.get("table_image_ids", [])]
            if sum(table_length) > 1000:
                result = {
                    "question": row.get("question"),
                    "golden_answer": row.get("answer"),
                    "table_image_ids": row.get("table_image_ids"),
                    "response": "Table too large to process.",
                }
                out_file.write(json.dumps(result) + "\n")
                continue
            
            
            try:
                inputs = prepare_phi3_input(row, images_dir, processor)
                response = generate_phi3_response(model, processor, inputs)
            
            except Exception as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    result = {
                        "question": row.get("question"),
                        "golden_answer": row.get("answer"),
                        "table_image_ids": row.get("table_image_ids"),
                        "response": "CUDA out of memory. Skipping.",
                    }
                    out_file.write(json.dumps(result) + "\n")
                    continue
                else:
                    
                    result = {
                        "question": row.get("question"),
                        "golden_answer": row.get("answer"),
                        "table_image_ids": row.get("table_image_ids"),
                        "response": "CUDA out of memory. Skipping.",
                    }
                    out_file.write(json.dumps(result) + "\n")
                    continue
                    

            result = {
                "question": row.get("question"),
                "golden_answer": row.get("answer"),
                "table_image_ids": row.get("table_image_ids"),
                "response": response,
            }
            out_file.write(json.dumps(result) + "\n")
            if (i + 1) % 10 == 0:
                out_file.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing JSONL and table_images.")
    parser.add_argument("--output-file", default="results/phi3_two_table_eval.jsonl")
    parser.add_argument("--model-path", default="microsoft/Phi-3.5-vision-instruct")
    parser.add_argument("--device", default="cuda:3", help="Device to run the model on")
    args = parser.parse_args()

    jsonl_path = find_jsonl_file(args.input_dir)
    if not jsonl_path:
        raise FileNotFoundError("No .jsonl file found in the specified directory.")
    data = benchmark_data(jsonl_path)

    images_dir = os.path.join(args.input_dir, "table_images")

    print(f"Jsonl file: {jsonl_path}")
    print(f"Images directory: {images_dir}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"No directory named 'table_images' found in {args.input_dir}.")
    
    evaluate_phi3(data, args.output_file, args.model_path, images_dir, args.device)

if __name__ == "__main__":
    main()
    
