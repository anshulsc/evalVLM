""" python phi3_eval.py --input-dir ../DataVLM/mmqa_extended/two_tables \
    --output-file results/phi3_two_table_eval.jsonl \
    --model-path microsoft/Phi-3.5-vision-instruct \
    --device cuda:1 \
    --quantization True \
    --quant_bit 4
"""

import os
import tqdm
import json
import argparse
import torch
import outlines
from PIL import Image
from transformers import  AutoProcessor, BitsAndBytesConfig, Phi3ForCausalLM, AutoModelForCausalLM  

from utils import find_jsonl_file, benchmark_data, SYSTEMS_INSTRUCTIONS_2, Response

def load_phi3_model(model_path, device="cuda:3", quantization=False, quant_bit=4):
    
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
            model_class = AutoModelForCausalLM,
            model_kwargs = {
                "quantization_config": bnb_config,
                "device_map": {"": device},
                'torch_dtype': torch.float16,
                'attn_implementation': 'eager',
                'trust_remote_code': True
                
                },
            processor_kwargs={
                "device": device,
                }
            )
        
    else:
        print("Quantization disabled")
        model = outlines.models.transformers_vision(
        model_path,
        model_class = AutoModelForCausalLM,
        model_kwargs = {
            "device_map": {"": device},
            'torch_dtype': torch.float16,
            'attn_implementation': 'eager',
            'trust_remote_code': True
        },
        
        processor_kwargs={
            "device": device,
        }
    )
    
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     device_map={"": device},
    #     trust_remote_code=True,
    #     quantization_config=bnb_config,
    #     _attn_implementation = 'eager'
    # )
    # model.eval()
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor

def prepare_phi3_input(data_row, images_dir, processor):
    images = []
    for image_id in data_row["table_image_ids"]:
        image_path = os.path.join(images_dir, image_id)
        images.append(Image.open(image_path))
    
    # Phi-3 uses special image placeholders
    image_placeholders = "\n".join([f"<|image_{i+1}|>" for i in range(len(images))])
    prompt = f"\n Please respond in JSON format as mention above. \n\n Question: {data_row['question']}\n answer:"
    
    messages = [{
        "role": "user", 
        "content": f"{SYSTEMS_INSTRUCTIONS_2}{Response}\n\n{image_placeholders}\n\n{prompt}"
    }]
    
    inputs = processor.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
  
    return (inputs, images)

def generate_phi3_response(model, processor, inputs):
    
    with torch.inference_mode():
        image_data_generator = outlines.generate.json(model, 
                                                     Response)
        image_data = image_data_generator(inputs[0], inputs[1])
        image_data = image_data.dict()
        print(f"Image data: {image_data}")
        
        return image_data

def evaluate_phi3(data, output_file, model_path, images_dir, device="cuda:3", quantization=False, quant_bit=4):
    model, processor = load_phi3_model(model_path, device, quantization, quant_bit)
    print("Evaluation Started....")

    with open(output_file, "a+", encoding="utf-8") as out_file:
        for i, row in enumerate(tqdm.tqdm(data)):
      
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
    
    evaluate_phi3(data, args.output_file, args.model_path, images_dir, args.device, args.quantization, args.quant_bit)

if __name__ == "__main__":
    main()
    
