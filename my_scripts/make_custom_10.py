import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_BASE"),
)

model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-v3.2")

prompt = """
请生成10条用于知识编辑实验的英文JSON数据。

每条数据必须包含：
- prompt: 补全式英文提示，例如 "The capital of Australia is"
- target_new: 希望编辑后的新答案，短实体
- ground_truth: 原始正确答案或模型可能已有答案
- rephrase_prompt: prompt的英文改写问法
- locality_prompt: 与该编辑无关的英文补全式事实
- locality_ground_truth: locality_prompt的正确答案

要求：
1. 输出必须是JSON数组。
2. 不要输出Markdown。
3. 每个答案尽量是人名、地名、组织名等短实体。
4. 可以使用反事实编辑，例如把 Australia's capital 从 Canberra 编辑为 Sydney。
"""

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You generate clean JSON datasets."},
        {"role": "user", "content": prompt},
    ],
    temperature=0.2,
)

content = response.choices[0].message.content.strip()

os.makedirs("data", exist_ok=True)

try:
    data = json.loads(content)
except json.JSONDecodeError:
    print(content)
    raise

with open("data/custom_10.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Saved data/custom_10.json")
print(json.dumps(data[:2], ensure_ascii=False, indent=2))