import os
from openai import OpenAI
from api.prompt import Prompt

class ChatGPT:
    def __init__(self):
        self.prompt = Prompt()
        # 設定 OpenAI API 金鑰
        self.api_key = os.getenv("OPENAI_API_KEY")

        # 初始化 OpenAI 客戶端
        client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
            base_url = os.getenv("OPENAI_BASE_URL", default="https://free.v36.cm/v1") 
        )


        # 設定其他參數
        self.model = os.getenv("OPENAI_MODEL", default="gpt-4")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", default=0))
        self.frequency_penalty = float(os.getenv("OPENAI_FREQUENCY_PENALTY", default=0))
        self.presence_penalty = float(os.getenv("OPENAI_PRESENCE_PENALTY", default=0.6))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", default=120))

    def get_response(self):
        try:
            # 使用 OpenAI SDK 呼叫 API
            response = client.chat.completions.create(
                model=self.model,
                prompt=self.prompt.generate_prompt(),
                temperature=self.temperature,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                max_tokens=self.max_tokens
            )
            # 回傳文字結果
            return response.choices[0].text.strip()
        except openai.error.OpenAIError as e:
            # 捕捉 API 錯誤並回傳錯誤訊息
            return f"抱歉，目前無法處理您的請求。錯誤訊息：{str(e)}"

    def add_msg(self, text):
        self.prompt.add_msg(text)