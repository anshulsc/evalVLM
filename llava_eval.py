"""
python llava_eval.py --input-dir ../DataVLM/mmqa_extended/two_tables \
    --output-file results/llava_two_table_eval.jsonl \
    --model-path llava-hf/llava-onevision-qwen2-7b-ov-hf \
    --device cuda:0
"""



import gc
import os

import tqdm
import json
import argparse
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration , BitsAndBytesConfig
from utils import find_jsonl_file, benchmark_data, SYSTEMS_INSTRUCTIONS

def load_llava_model(model_path, device="cuda:3"):
    
    bnb_config = BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_quant_type="nf4",
          bnb_4bit_use_double_quant=True,
          bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": device}
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def prepare_llava_input(data_row, images_dir, processor):
    conversation = [{
        "role": "user",
        "content": []
    }]

    # Add images first
    for image_id in data_row["table_image_ids"]:
        image_path = os.path.join(images_dir, image_id)
        conversation[0]["content"].append({
            "type": "image",
        })

    # Add text prompt
    conversation[0]["content"].append({
        "type": "text",
        "text": f"{SYSTEMS_INSTRUCTIONS}\n\nQuestion: {data_row['question']}\nAnswer:"
    })

    # Process inputs
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        text=prompt,
        images=[Image.open(os.path.join(images_dir, img_id)) for img_id in data_row["table_image_ids"]],
        return_tensors="pt"
    )
    return inputs

def generate_llava_response(model, processor, inputs):
    with torch.inference_mode():
        inputs = inputs.to(model.device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        output = processor.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return output

def evaluate_llava(data, output_file, model_path, images_dir, device="cuda:0"):
    model, processor = load_llava_model(model_path, device)
    
    print("Evaluation Started...")
    with open(output_file, "a+", encoding="utf-8") as out_file:
        for i, row in enumerate(tqdm.tqdm(data)):
            # Skip large tables (adapt threshold as needed)
            table_length = [int(img.split("_")[-1].split(".")[0]) for img in row.get("table_image_ids", [])]
            if sum(table_length) > 1000:
                result = {
                    "question": row["question"],
                    "golden_answer": row["answer"],
                    "table_image_ids": row["table_image_ids"],
                    "response": "Table too large to process.",
                }
                out_file.write(json.dumps(result) + "\n")
                continue
            
            try:
                inputs = prepare_llava_input(row, images_dir, processor)
                response = generate_llava_response(model, processor, inputs)
            except RuntimeError as e:
                print(f"Error processing question {row['question']}: {e}")
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    result = {
                        "question": row["question"],
                        "golden_answer": row["answer"],
                        "table_image_ids": row["table_image_ids"],
                        "response": f"{e}",
                    }
                    out_file.write(json.dumps(result) + "\n")
                    continue
                else:
                    raise e

            result = {
                "question": row["question"],
                "golden_answer": row["answer"],
                "table_image_ids": row["table_image_ids"],
                "response": response,
            }
            out_file.write(json.dumps(result) + "\n")
            
            if (i + 1) % 10 == 0:
                out_file.flush()

def main():
    torch.cuda.empty_cache()
    gc.collect()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing JSONL and table_images")
    parser.add_argument("--output-file", default="results/llava_two_table_eval.jsonl")
    parser.add_argument("--model-path", default="llava-hf/llava-onevision-qwen2-7b-ov-hf")
    parser.add_argument("--device", default="cuda:0", help="Device to run the model on")
    args = parser.parse_args()

    jsonl_path = find_jsonl_file(args.input_dir)
    data = benchmark_data(jsonl_path)
    images_dir = os.path.join(args.input_dir, "table_images")

    evaluate_llava(data, args.output_file, args.model_path, images_dir, args.device)

if __name__ == "__main__":
    main()