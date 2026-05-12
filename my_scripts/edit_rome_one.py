import os
import json
import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", default="my_hparams/ROME_qwen2.5_0.5b.yaml")
    parser.add_argument("--data", default="data/custom_10.json")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", default="outputs/rome_one.json")
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    item = data[args.index]

    hparams = ROMEHyperParams.from_hparams(args.hparams)
    editor = BaseEditor.from_hparams(hparams)
    editor.model.config.use_cache = False

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

    result = {
        **item,
        "direct_output_after": direct_output,
        "rephrase_output_after": rephrase_output,
        "locality_output_after": locality_output,
        "ES_hit": hit(direct_output, item["target_new"]),
        "PS_hit": hit(rephrase_output, item["target_new"]),
        "NS_hit": hit(locality_output, item["locality_ground_truth"]),
        "easyedit_metrics": metrics,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
    