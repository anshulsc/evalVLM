
""" python qwen_eval.py --input-dir ../DataVLM/mmqa_extended/two_tables \
    --output-file results/qwen_two_table_eval.jsonl \
    --model-path Qwen/Qwen2-VL-2B-Instruct \
    --device cuda:1
    --quantization False    
"""
import gc
import os
import tqdm
import json
import argparse
import time
from pydantic import BaseModel
from typing import List, Optional
import outlines 
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image

from utils import find_jsonl_file, benchmark_data, SYSTEMS_INSTRUCTIONS_2, Response

def load_qwen_model(model_path, device="cuda:1", quantization=False, quant_bit=4):
    
    if quantization:
        
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
            model_class = Qwen2VLForConditionalGeneration,
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
        model_class = Qwen2VLForConditionalGeneration,
        model_kwargs = {
            "device_map": {"": device},
            'torch_dtype': torch.float16
        },
        
        processor_kwargs={
            "device": device,
        }
    )
    
        # model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     model_path, 
        #     quantization_config = bnb_config,
        #     device_map = {"":device}
        # )
        # model.eval()
        
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def prepare_qwen_input(data_row, images_dir, processor):
    
    messages = [
        
        {
            "role": "user",
            "content": []
        }
    ]
    
    
    
    images = []
    for image_id in data_row["table_image_ids"]:
        image_path = os.path.join(images_dir, image_id)
        messages[0]["content"].append(
            {
                "type": "image",
                "image": Image.open(image_path),
            }
        )
        images.append(Image.open(image_path))


    messages[0]["content"].append(
        {
            "type": "text",
            "text": f"{SYSTEMS_INSTRUCTIONS_2}{Response.model_json_schema()} \n\n question:{data_row['question']} \n answer:",
        }
    )
    
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # image_inputs, Video = process_vision_info(messages)
    # inputs = processor(
    #     text=[text],
    #     images=image_inputs,
    #     padding=True,
    #     return_tensors="pt",
    # )
    
    return (text, images)

def generate_qwen_response(model, processor, inputs):
         
    with torch.inference_mode():
        
        image_data_generator = outlines.generate.json(model, 
                                                     Response)
        
        image_data = image_data_generator(inputs[0], inputs[1])
        image_data = image_data.dict()
        
        # generated_ids = model.generate(**inputs, max_new_tokens=500)
        # generated_ids_trimmed = [
        #     out_ids[len(in_ids):]
        #     for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        # ]
        # output_text = processor.batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )
        return image_data

def evaluate_qwen(data, output_file, model_path, images_dir, device="cuda:3", quantization=False, quant_bit=4):
    
    model, processor = load_qwen_model(model_path, device, quantization, quant_bit)
    
    print("Evaluation Sarted....")

    with open(output_file, "a+", encoding="utf-8") as out_file:
        
        for i, row in enumerate(tqdm.tqdm(data)):
            
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
                inputs = prepare_qwen_input(row, images_dir, processor)
                response = generate_qwen_response(model, processor, inputs)
            except RuntimeError as e:
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
                    raise e

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
    
    torch.cuda.empty_cache()
    gc.collect()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input-dir", required=True, help="Directory containing JSONL and table_images.")
    parser.add_argument("--output-file", default="results/qwen_two_table_eval.jsonl")
    parser.add_argument("--model-path", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--device", default="cuda:3", help="Device to run the model on, e.g., 'cuda:0', 'cpu'.")
    parser.add_argument("--quantization", default=False, help="Quantization flag")
    parser.add_argument("--quant_bit", default=4, choices=[4,8], help="Quantization bit", type=int)
    
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
    
    evaluate_qwen(data, args.output_file, args.model_path, images_dir, args.device, args.quantization, args.quant_bit)

if __name__ == "__main__":
    main()
    
