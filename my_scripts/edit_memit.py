import os
import json
import time
import argparse
from tqdm import tqdm

import torch

from easyeditor import BaseEditor
from easyeditor import MEMITHyperParams


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


def peak_gpu_memory_gb():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", default="my_hparams/MEMIT_gpt2_xl.yaml")
    parser.add_argument("--data", default="data/custom_10_with_subject.json")
    parser.add_argument("--out", default="outputs/memit_debug_gpt2xl.json")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)[:args.limit]

    prompts = [x["prompt"] for x in data]
    subjects = [x["subject"] for x in data]
    ground_truth = [x["ground_truth"] for x in data]
    target_new = [x["target_new"] for x in data]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("[1/6] Loading MEMIT hparams...", flush=True)
    hparams = MEMITHyperParams.from_hparams(args.hparams)

    print("[2/6] Instantiating BaseEditor/model...", flush=True)
    editor = BaseEditor.from_hparams(hparams)

    print("[3/6] Starting MEMIT edit...", flush=True)
    start = time.time()

    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        subject=subjects,
        ground_truth=ground_truth,
        target_new=target_new,
        keep_original_weight=False,
    )

    print("[4/6] MEMIT edit finished.", flush=True)
    elapsed = time.time() - start
    peak_mem = peak_gpu_memory_gb()

    tokenizer = editor.tok

    results = []
    print("[5/6] Starting generation-based evaluation...", flush=True)
    for idx, item in enumerate(tqdm(data)):
        direct_output = generate(edited_model, tokenizer, item["prompt"])
        rephrase_output = generate(edited_model, tokenizer, item["rephrase_prompt"])
        locality_output = generate(edited_model, tokenizer, item["locality_prompt"])

        row = {
            **item,
            "direct_output_after": direct_output,
            "rephrase_output_after": rephrase_output,
            "locality_output_after": locality_output,
            "ES_hit": hit(direct_output, item["target_new"]),
            "PS_hit": hit(rephrase_output, item["target_new"]),
            "NS_hit": hit(locality_output, item["locality_ground_truth"]),
        }

        results.append(row)

        print("=" * 80)
        print(f"MEMIT Sample {idx}")
        print("PROMPT:", item["prompt"])
        print("TARGET_NEW:", item["target_new"])
        print("DIRECT_AFTER:", direct_output)
        print("REPHRASE_AFTER:", rephrase_output)
        print("LOCALITY_AFTER:", locality_output)
        print("ES:", row["ES_hit"], "PS:", row["PS_hit"], "NS:", row["NS_hit"])

    summary = {
        "num_edits": len(data),
        "time_seconds": elapsed,
        "peak_gpu_memory_gb": peak_mem,
        "easyedit_metrics": metrics,
        "results": results,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("MEMIT num_edits:", len(data))
    print("MEMIT time_seconds:", elapsed)
    print("MEMIT peak_gpu_memory_gb:", peak_mem)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()