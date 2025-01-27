""" 
python evaluate_metrics.py \
  --input-file results/qwen_two_table_eval.jsonl \
  --output-file results/exp/qwen_two_table_metrics.json \
  --wandb-project evalVLM2 \
  --wandb-entity aurelius_ \
  --wandb-run-name exp
  
"""

import os
import json
import argparse
from collections import Counter
from typing import Dict, List, Union
import tqdm
import wandb 
import re

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

def normalize_element(element: Union[str, int, float, list]) -> Union[str, int, float, list]:
    if isinstance(element, dict):
        values = [element[key] for key in element.keys()]
        return normalize_item(values)
    if isinstance(element, str):
        element = re.sub(r'[{}]|[\w-]+:', '', element)
        return normalize_item(element)
    if isinstance(element, list):
        return [normalize_item(e) for e in element]
    return normalize_item(element)

def normalize_item(item: Union[str, int, float]) -> Union[str, int, float]:
    if isinstance(item, list):
        return [normalize_item(e) for e in item]
    if isinstance(item, (int, float)):
        return int(item) if isinstance(item, float) and item.is_integer() else item
    if isinstance(item, str):
        clean_str = item.replace(",", "").strip()
        if clean_str.replace('.', '', 1).isdigit():
            num = float(clean_str)
            return int(num) if num.is_integer() else round(num, 6)
        return clean_str.lower()
    return item

def structure_to_string(data: Union[list, dict, str]) -> str:
    original_data = data
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            # Attempt to fix common JSON errors (e.g., unquoted keys)
            fixed_str = re.sub(r'([{,]\s*)([a-zA-Z_]\w*)(\s*:)', lambda m: f'{m.group(1)}"{m.group(2)}"{m.group(3)}', data)
            try:
                data = json.loads(fixed_str)
            except json.JSONDecodeError:
                # Fallback: extract numbers and words from the string
                numbers = re.findall(r'\b\d+\b', data)
                words = re.findall(r'\b[a-zA-Z]+\b', data)
                data = numbers + words

    if isinstance(data, dict):
        data = data.get("data", [])
    
    normalized = normalize_element(data)
    
    def flatten(items):
        result = []
        for item in items:
            if isinstance(item, list):
                result.extend(flatten(item))
            else:
                result.append(str(item))
        return result
    
    flattened = flatten(normalized) if not isinstance(normalized, (str, int, float)) else [str(normalized)]
    return " ".join(flattened)

def compute_exact_match(prediction: str, truth: str) -> int:
    return int(prediction == truth)

def compute_f1(prediction: str, truth: str) -> float:
    pred_tokens = prediction.split()
    truth_tokens = truth.split()
    
    if not pred_tokens or not truth_tokens:
        return int(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    overlap = sum(common.values())
    
    if overlap == 0:
        return 0.0
    
    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

def process_jsonl(input_file: str) -> Dict:
    results = []
    total = 0
    total_skipped = 0
    parse_errors = 0
    cuda_errors = 0
    table_errors = 0
    exact_matches = 0
    f1_sum = 0.0
    
    i = 0
    with open(input_file, "r") as f:
        for line in tqdm.tqdm(f, desc=f"{Colors.OKBLUE}Processing entries{Colors.RESET}"):
            if i == 51:
                break
            line = line.strip()
            if not line:
                continue
                
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                parse_errors += 1
                print(f"{Colors.FAIL}JSON parse error:{Colors.RESET} {str(e)}")
                continue
            
            if "response" in entry and any(err in entry["response"] for err in ["CUDA out of memory", "Table too large"]):
                results.append({
                    "question": entry.get('question'),
                    "golden_answer": str(entry.get('golden_answer', {}).get('data', '')),
                    "skipped": True,
                    "skip_reason": entry["response"]
                })
                total_skipped += 1
                if "CUDA out of memory" in entry["response"]:
                    cuda_errors += 1
                if "Table too large" in entry["response"]:
                    table_errors += 1
                continue
            
            total += 1
            
            try:
                model_response = entry.get("response", {})
                if isinstance(model_response, str):
                    model_response = json.loads(model_response)
            except json.JSONDecodeError:
                model_response = {"data": []}
            
            model_str = structure_to_string(model_response).lower()
            gold_str = structure_to_string(entry.get("golden_answer", {})).lower()
            
            em = compute_exact_match(model_str, gold_str)
            f1 = compute_f1(model_str, gold_str)
            
            exact_matches += em
            f1_sum += f1
            
            results.append({
                "question": entry.get('question'),
                "golden_answer": str(entry.get('golden_answer', {}).get('data', '')),
                "model_response": str(entry.get('response')),
                "golden_answer_str": gold_str,
                "model_response_str": model_str,
                "exact_match": em,
                "f1_score": f1,
                "skipped": False
            })
            i+= 1 
    
    return {
        "overall_metrics": {
            "exact_match": exact_matches / total if total > 0 else 0,
            "f1_score": f1_sum / total if total > 0 else 0,
            "processed_samples": total,
            "skipped_samples": total_skipped,
            "parse_errors": parse_errors,
            "cuda_errors": cuda_errors,
            "table_errors": table_errors,
            "total_samples": total + total_skipped + parse_errors
        },
        "per_sample_results": results
    }

def main():
    parser = argparse.ArgumentParser(description="Calculate evaluation metrics from model results")
    parser.add_argument("--input-file", required=True, help="Path to JSONL file with model responses")
    parser.add_argument("--output-file", required=True, help="Path to save JSON metrics file")
    parser.add_argument("--wandb-project", help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", help="Weights & Biases entity")
    parser.add_argument("--wandb-run-name", help="Weights & Biases run name")
    
    args = parser.parse_args()
    
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    print(f"{Colors.OKBLUE}Processing results from:{Colors.RESET} {Colors.BOLD}{args.input_file}{Colors.RESET}")
    metrics = process_jsonl(args.input_file)
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    if args.wandb_project:
        wandb.log(metrics["overall_metrics"])
    
    print(f"\n{Colors.OKGREEN}Evaluation results saved to:{Colors.RESET} {Colors.OKCYAN}{args.output_file}{Colors.RESET}")
    print(f"{Colors.OKGREEN}Processed Samples:{Colors.RESET} {Colors.OKBLUE}{metrics['overall_metrics']['processed_samples']}{Colors.RESET}")
    print(f"{Colors.WARNING}Skipped Samples:{Colors.RESET}  {Colors.OKBLUE}{metrics['overall_metrics']['skipped_samples']}{Colors.RESET}")
    print(f"{Colors.FAIL}Parse Errors:{Colors.RESET}       {Colors.OKBLUE}{metrics['overall_metrics']['parse_errors']}{Colors.RESET}")
    
    em = metrics['overall_metrics']['exact_match']
    f1 = metrics['overall_metrics']['f1_score']
    em_color = Colors.OKGREEN if em > 0.5 else Colors.WARNING
    f1_color = Colors.OKGREEN if f1 > 0.6 else Colors.WARNING
    
    print(f"{Colors.OKGREEN}Exact Match (%):{Colors.RESET}    {em_color}{em * 100:.2f}{Colors.RESET}")
    print(f"{Colors.OKGREEN}Average F1 Score (%):{Colors.RESET} {f1_color}{f1 * 100:.2f}{Colors.RESET}")

if __name__ == "__main__":
    main()