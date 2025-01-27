"""
python llava_eval.py --input-dir ../DataVLM/mmqa_extended/two_tables \
    --output-file results/llava_two_table_eval.jsonl \
    --model-path llava-hf/llava-onevision-qwen2-7b-ov-hf \
    --device cuda:1 \
    --quantization True \
    --quant_bit 4
    
"""



import gc
import os

import tqdm
import json
import argparse
import requests
import torch
import outlines

from PIL import Image
from transformers import AutoProcessor,  LlavaOnevisionForConditionalGeneration , BitsAndBytesConfig
from utils import find_jsonl_file, benchmark_data, SYSTEMS_INSTRUCTIONS_2, Response



        
        
def load_llava_model(model_path, device="cuda:3", quantization=False, quant_bit=4):
    
    if quantization:
        print("Quantization enabled")
        if quant_bit == 4:
            bnb_config =BitsAndBytesConfig(
                  load_in_4bit=True,
                  bnb_4bit_quant_type="nf4",
                  bnb_4bit_use_double_quant=True,
                  bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config =BitsAndBytesConfig(
                    load_in_8bit=True,
            )
            
        model = outlines.models.transformers_vision(
            model_path,
            model_class = LlavaOnevisionForConditionalGeneration,
            model_kwargs = {
                "quantization_config": bnb_config,
                "device_map": {"": device},
                'torch_dtype': torch.float16
                },
            processor_kwargs={
                "device": device,
                }
            )
        
    else:
        model = outlines.models.transformers_vision(
        model_path,
        model_class = LlavaOnevisionForConditionalGeneration,
        model_kwargs = {
            "device_map": {"": device},
            'torch_dtype': torch.float16
        },
        
        processor_kwargs={
            "device": device,
        }
    )
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
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
        "text": f"{SYSTEMS_INSTRUCTIONS_2}{Response.model_json_schema()}\n\nQuestion: {data_row['question']}\nAnswer:"
    })

    # Process inputs
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    images = [Image.open(os.path.join(images_dir, img_id)) for img_id in data_row["table_image_ids"]]
    
    return (prompt, images)

def generate_llava_response(model, processor, inputs):
    
    with torch.inference_mode():
        image_data_generator = outlines.generate.json(model, 
                                                     Response) 
        image_data = image_data_generator(inputs[0], inputs[1])
        image_data = image_data.dict()
        print(image_data)
        return image_data

def evaluate_llava(data, output_file, model_path, images_dir, device="cuda:0", quantization=False, quant_bit=4):
    model, processor = load_llava_model(model_path, device, quantization, quant_bit)
    
    print("Evaluation Started...")
    with open(output_file, "a+", encoding="utf-8") as out_file:
        for i, row in enumerate(tqdm.tqdm(data)):
            
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
    parser.add_argument("--quantization", default=False, help="Quantization flag")
    parser.add_argument("--quant_bit", default=4, choices=[4,8], help="Quantization bit", type=int)
    args = parser.parse_args()

    jsonl_path = find_jsonl_file(args.input_dir)
    data = benchmark_data(jsonl_path)
    images_dir = os.path.join(args.input_dir, "table_images")

    evaluate_llava(data, args.output_file, args.model_path, images_dir, args.device, args.quantization, args.quant_bit)

if __name__ == "__main__":
    main()