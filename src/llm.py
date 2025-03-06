# 标准库导入
import json
import requests

# 第三方库导入
from openai import OpenAI

# 本地模块导入
from logger import LOG

class LLM:
    """LLM 类用于处理与大语言模型的交互，支持 OpenAI 和 Ollama 两种模型。"""

    def __init__(self, config):
        """
        初始化 LLM 实例。

        Args:
            config: 配置对象，包含模型配置参数
        
        Raises:
            ValueError: 当模型类型不支持时抛出
        """
        self.config = config
        self.model = config.llm_model_type.lower()

        if self.model == "openai":
            self.client = OpenAI()
        elif self.model == "ollama":
            self.api_url = config.ollama_api_url
        else:
            error_msg = f"不支持的模型类型: {self.model}"
            LOG.error(error_msg)
            raise ValueError(error_msg)

    def generate_report(self, system_prompt, user_content):
        """
        根据系统提示和用户内容生成报告。

        Args:
            system_prompt (str): 系统提示信息
            user_content (str): 用户提供的内容

        Returns:
            str: 生成的报告内容

        Raises:
            ValueError: 当模型类型不支持时抛出
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        if self.model == "openai":
            return self._generate_report_openai(messages)
        elif self.model == "ollama":
            return self._generate_report_ollama(messages)
        
        raise ValueError(f"不支持的模型类型: {self.model}")

    def _generate_report_openai(self, messages):
        """
        使用 OpenAI 模型生成报告。

        Args:
            messages (list): 消息列表

        Returns:
            str: 生成的报告内容

        Raises:
            Exception: API 调用失败时抛出
        """
        LOG.info(f"使用 OpenAI {self.config.openai_model_name} 模型生成报告")
        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model_name,
                messages=messages
            )
            LOG.debug("GPT 响应: %s", response)
            return response.choices[0].message.content
        except Exception as e:
            LOG.error("OpenAI 生成报告失败: %s", str(e))
            raise

    def _generate_report_ollama(self, messages):
        """
        使用 Ollama 模型生成报告。

        Args:
            messages (list): 消息列表

        Returns:
            str: 生成的报告内容

        Raises:
            ValueError: API 返回无效响应时抛出
            Exception: API 调用失败时抛出
        """
        LOG.info(f"使用 Ollama {self.config.ollama_model_name} 模型生成报告")
        try:
            payload = {
                "model": self.config.ollama_model_name,
                "messages": messages,
                "max_tokens": 4000,
                "temperature": 0.7,
                "stream": False
            }

            response = requests.post(self.api_url, json=payload)
            response_data = response.json()
            LOG.debug("Ollama 响应: %s", response_data)

            message_content = response_data.get("message", {}).get("content")
            if not message_content:
                raise ValueError("Ollama API 返回的响应结构无效")
            
            return message_content
        except Exception as e:
            LOG.error("Ollama 生成报告失败: %s", str(e))
            raise

if __name__ == '__main__':
    from config import Config
    
    # 测试代码
    config = Config()
    llm = LLM(config)

    markdown_content = """
# Progress for langchain-ai/langchain (2024-08-20 to 2024-08-21)

## Issues Closed in the Last 1 Days
- partners/chroma: release 0.1.3 #25599
- docs: few-shot conceptual guide #25596
- docs: update examples in api ref #25589
"""

    system_prompt = (
        "你是资深的数据分析师，根据下面最新的项目进展，清晰地划分不同类型的内容，"
        "生成一份简报，以便项目后续的跟进与评估，类型至少包括："
        "1）概况；2）新增功能；3）功能优化；4）问题修复"
    )
    
    github_report = llm.generate_report(system_prompt, markdown_content)
    LOG.debug(github_report)
