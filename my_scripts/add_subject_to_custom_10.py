import json
import re


def guess_subject(prompt: str) -> str:
    patterns = [
        r"The capital of (.*?) is",
        r"The author of (.*?) is",
        r"The CEO of (.*?) is",
        r"The current CEO of (.*?) is",
        r"The president of (.*?) is",
        r"The current president of (.*?) is",
        r"The founder of (.*?) is",
        r"The owner of (.*?) is",
        r"The headquarters of (.*?) is",
        r"The mother tongue of (.*?) is",
        r"The official language of (.*?) is",

        # 处理这种结构：The current UN Secretary-General is
        r"The current (.*?) is",

        # 处理这种结构：The UN Secretary-General is
        r"The (.*?) is",
    ]

    for pattern in patterns:
        m = re.search(pattern, prompt)
        if m:
            return m.group(1).strip()

    # fallback：尽量取 of 后面、is 前面的内容
    m = re.search(r"of (.*?) is", prompt)
    if m:
        return m.group(1).strip()

    raise ValueError(f"Cannot guess subject from prompt: {prompt}")


input_path = "data/custom_10.json"
output_path = "data/custom_10_with_subject.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    if "subject" not in item or not item["subject"]:
        item["subject"] = guess_subject(item["prompt"])

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Saved {output_path}")
for i, item in enumerate(data):
    print(i, "| prompt:", item["prompt"], "| subject:", item["subject"])