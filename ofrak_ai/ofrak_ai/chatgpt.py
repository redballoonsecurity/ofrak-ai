import os
import openai

from dataclasses import dataclass
from openai.openai_object import OpenAIObject
from openai.error import OpenAIError
from typing import List, Dict, Optional

from ofrak.model.component_model import ComponentConfig
from ofrak.model.resource_model import ResourceAttributes
from ofrak_ai.exponential_backoff import retry_with_exponential_backoff


@dataclass
class ChatGPTConfig(ComponentConfig):
    """
    :param api_key: the OpenAI API key to use
    :param api_organization: the OpenAI API organization to use
    :param model: the OpenAI model to use
    :param system_message: a message which can be prepended to the conversation in API calls, which
        is used for providing additional information to ChatGPT for generating responses
    :param temperature: a measure of the randomness of ChatGPT's responses, between 0 (low) and
        2 (high)
    """

    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    api_organization: Optional[str] = os.getenv("OPENAI_ORGANIZATION")
    model: str = "gpt-3.5-turbo"
    system_message: Optional[str] = None
    temperature: float = 1.0


@dataclass
class ChatGPTAnalysis(ResourceAttributes):
    description: str


async def get_chatgpt_response(
    history: List[Dict[str, str]],
    max_tokens: int,
    config: ChatGPTConfig,
) -> Optional[OpenAIObject]:
    @retry_with_exponential_backoff
    async def retry_response(**kwargs) -> Optional[str]:
        try:
            response = await openai.ChatCompletion.acreate(**kwargs)
            return response
        except OpenAIError as e:
            raise e

    return await retry_response(
        model=config.model,
        temperature=config.temperature,
        max_tokens=max_tokens,
        messages=[message for message in history],
    )
