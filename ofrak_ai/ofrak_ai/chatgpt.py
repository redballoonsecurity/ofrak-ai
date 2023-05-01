import os

from dataclasses import dataclass
from typing import Optional

from ofrak.model.component_model import ComponentConfig
from ofrak.model.resource_model import ResourceAttributes


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
    temperature: float = 1


@dataclass
class ChatGPTAnalysis(ResourceAttributes):
    description: str
