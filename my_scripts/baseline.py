import os
import json
import argparse
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate(model, tokenizer, prompt, max_new_tokens=8):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if full_text.startswith(prompt):
        return full_text[len(prompt):].strip()

    return full_text.strip()


def hit(output, answer):
    return answer.lower().strip() in output.lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/gpt2-xl")
    parser.add_argument("--data", default="data/custom_10_with_subject.json")
    parser.add_argument("--out", default="outputs/baseline_gpt2xl.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    model.eval()

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for idx, item in enumerate(tqdm(data)):
        direct_output = generate(model, tokenizer, item["prompt"])
        rephrase_output = generate(model, tokenizer, item["rephrase_prompt"])
        locality_output = generate(model, tokenizer, item["locality_prompt"])

        row = {
            **item,
            "direct_output_before": direct_output,
            "rephrase_output_before": rephrase_output,
            "locality_output_before": locality_output,
            "target_new_in_direct_before": hit(direct_output, item["target_new"]),
            "ground_truth_in_direct_before": hit(direct_output, item["ground_truth"]),
        }

        results.append(row)

        print("=" * 80)
        print(f"Baseline Sample {idx}")
        print("PROMPT:", item["prompt"])
        print("TARGET_NEW:", item["target_new"])
        print("GROUND_TRUTH:", item["ground_truth"])
        print("BEFORE_DIRECT:", direct_output)
        print("BEFORE_REPHRASE:", rephrase_output)
        print("BEFORE_LOCALITY:", locality_output)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()