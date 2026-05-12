import os
import json
import time
import argparse
from tqdm import tqdm

import torch

from easyeditor import BaseEditor
from easyeditor import ROMEHyperParams


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


def easyedit_rewrite_acc(metrics):
    try:
        value = metrics[0]["post"]["rewrite_acc"]
        if isinstance(value, list):
            return float(value[0])
        return float(value)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", default="my_hparams/ROME_gpt2_xl.yaml")
    parser.add_argument("--data", default="data/custom_10_with_subject.json")
    parser.add_argument("--out", default="outputs/rome_results_gpt2xl.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    hparams = ROMEHyperParams.from_hparams(args.hparams)
    results = []

    for idx, item in enumerate(tqdm(data)):
        start = time.time()

        editor = BaseEditor.from_hparams(hparams)

        metrics, edited_model, _ = editor.edit(
            prompts=[item["prompt"]],
            subject=[item["subject"]],
            ground_truth=[item["ground_truth"]],
            target_new=[item["target_new"]],
            keep_original_weight=False,
        )

        tokenizer = editor.tok

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
            "easyedit_rewrite_acc": easyedit_rewrite_acc(metrics),
            "time_seconds": time.time() - start,
            "easyedit_metrics": metrics,
        }

        results.append(row)

        print("=" * 80)
        print(f"ROME Sample {idx}")
        print("PROMPT:", item["prompt"])
        print("SUBJECT:", item["subject"])
        print("TARGET_NEW:", item["target_new"])
        print("DIRECT_AFTER:", direct_output)
        print("REPHRASE_AFTER:", rephrase_output)
        print("LOCALITY_AFTER:", locality_output)
        print("ES:", row["ES_hit"], "PS:", row["PS_hit"], "NS:", row["NS_hit"])
        print("EasyEdit rewrite_acc:", row["easyedit_rewrite_acc"])

        del edited_model
        del editor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()