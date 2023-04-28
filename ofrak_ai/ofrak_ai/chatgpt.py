import os

from dataclasses import dataclass
from typing import Optional

from ofrak.model.component_model import ComponentConfig
from ofrak.model.resource_model import ResourceAttributes


@dataclass
class ChatGPTConfig(ComponentConfig):
    api_key: str = os.getenv("OPENAI_API_KEY")
    api_organization: str = os.getenv("OPENAI_ORGANIZATION")
    model: str = "gpt-3.5-turbo"
    system_message: Optional[str] = None
    temperature: int = 1


@dataclass
class ChatGPTAnalysis(ResourceAttributes):
    description: str
