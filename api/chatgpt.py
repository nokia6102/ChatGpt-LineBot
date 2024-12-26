import os
import openai  # 確保正確導入 OpenAI 模組
from api.prompt import Prompt  # 假設 Prompt 類在 api.prompt 模組中

class ChatGPT:
    """
    ChatGPT 客戶端，用於管理對話上下文並生成 OpenAI 的回應。
    """
    def __init__(self):
        self.prompt = Prompt()  # 初始化對話上下文

        # 初始化 OpenAI 客戶端
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("環境變數 OPENAI_API_KEY 未設定，無法初始化 OpenAI 客戶端。")

        base_url = os.getenv("OPENAI_BASE_URL", default="https://free.v36.cm/v1")
        self.client = openai.ChatCompletion(  # 使用正確的 API 初始化
            api_key=api_key,
            base_url=base_url
        )

        # 設定模型參數
        self.model = os.getenv("OPENAI_MODEL", default="gpt-4")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", default=0))
        self.frequency_penalty = float(os.getenv("OPENAI_FREQUENCY_PENALTY", default=0))
        self.presence_penalty = float(os.getenv("OPENAI_PRESENCE_PENALTY", default=0.6))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", default=120))

    def get_response(self):
        """
        生成 ChatGPT 的回應。
        :return: 回應文字或錯誤訊息
        """
        try:
            # 呼叫 OpenAI API 生成回應
            response = self.client.create(
                model=self.model,
                messages=self.prompt.generate_prompt(),
                temperature=self.temperature,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                max_tokens=self.max_tokens
            )
            # 回傳第一個回應的文字
            return response["choices"][0]["message"]["content"].strip()
        except openai.OpenAIError as e:
            # 捕捉 API 錯誤並提供清晰的錯誤訊息
            return f"抱歉，目前無法處理您的請求。錯誤訊息：{str(e)}"

    def add_msg(self, text):
        """
        新增一則訊息至對話上下文。
        :param text: 使用者輸入的文字
        """
        self.prompt.add_msg(text)