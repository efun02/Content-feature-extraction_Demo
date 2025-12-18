import json
import time

import requests
import time
class LLMClient:
    def __init__(self, api_key, api_url, model_name, max_retries=3):
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self.max_retries = max_retries

    def analyze(self, prompt, timeout=60):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}  # 👈 关键！强制模型输出 JSON（仅支持部分模型）
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=timeout
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # 尝试解析 JSON（防御模型输出带 markdown 或解释）
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # 如果失败，尝试提取 ```json ... ``` 中的内容
                    import re
                    match = re.search(r"```(?:json)?\s*({.*?})\s*```", content, re.DOTALL)
                    if match:
                        return json.loads(match.group(1))
                    else:
                        raise ValueError(f"LLM 返回非 JSON 内容: {content[:200]}...")

            except (requests.RequestException, ValueError, KeyError) as e:
                print(f"LLM 调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise RuntimeError("LLM 分析失败，已达到最大重试次数") from e
                time.sleep(2 ** attempt)  # 指数退避

        raise RuntimeError("Unexpected error in LLMClient")
