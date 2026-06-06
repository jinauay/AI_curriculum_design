import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SJTU_API_KEY")
BASE_URL = os.getenv("SJTU_API_BASE", "https://models.sjtu.edu.cn/api/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

def get_explanation(text, label, confidence, top_evidence=None):
    prompt = f"""
推文: {text}
预测标签: {label} (置信度: {confidence:.2f})
证据: {top_evidence or '无'}
请用中文生成一段解释，说明为什么这个推文是谣言或非谣言。
"""
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个负责生成解释的谣言检测助手。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    res_json = response.json()
    explanation = res_json["choices"][0]["message"]["content"].strip()
    return explanation