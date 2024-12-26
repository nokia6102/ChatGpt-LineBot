from api.prompt import Prompt
import os
import requests

class OpenAI:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url

    def create_completion(self, model, prompt, temperature, frequency_penalty, presence_penalty, max_tokens):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens
        }
        try:
            response = requests.post(
                f"{self.base_url}/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            raise Exception("ChatGPT API 請求超時，請稍後再試。")
        except requests.exceptions.HTTPError as http_err:
            raise Exception(f"HTTP 錯誤: {http_err}")
        except requests.exceptions.RequestException as err:
            raise Exception(f"請求錯誤: {err}")

class ChatGPT:
    def __init__(self):
        self.prompt = Prompt()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", default="https://api.openai.com/v1")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = os.getenv("OPENAI_MODEL", default="gpt-4")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", default=0))
        self.frequency_penalty = float(os.getenv("OPENAI_FREQUENCY_PENALTY", default=0))
        self.presence_penalty = float(os.getenv("OPENAI_PRESENCE_PENALTY", default=0.6))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", default=120))

    def get_response(self):
        try:
            response = self.client.create_completion(
                model=self.model,
                prompt=self.prompt.generate_prompt(),
                temperature=self.temperature,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                max_tokens=self.max_tokens
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            return f"抱歉，目前無法處理您的請求。錯誤訊息：{str(e)}"

    def add_msg(self, text):
        self.prompt.add_msg(text)
