import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_API_BASE")
model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-v3.2")

if not api_key:
    raise ValueError("DEEPSEEK_API_KEY is missing in .env")

if not base_url:
    raise ValueError("DEEPSEEK_API_BASE is missing in .env")

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "请用一句话解释什么是知识编辑。"},
    ],
    temperature=0,
)

print(response.choices[0].message.content)