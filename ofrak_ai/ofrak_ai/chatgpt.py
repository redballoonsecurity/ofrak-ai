import os
import openai

from dataclasses import dataclass
from enum import Enum
from openai.openai_object import OpenAIObject
from openai.error import OpenAIError
from typing import List, Dict, Optional

from ofrak.model.component_model import ComponentConfig
from ofrak_ai.exponential_backoff import retry_with_exponential_backoff


class ModelType(str, Enum):
    """
    Models ending in 4-digit long sequences are snapshots of their respective models. These are
    static versions that receive no updates and will be deprecated 3 months after the newest
    version is released.

    GPT-4-32k models have a max token value of 32,768 (4x higher than standard GPT-4).

    GPT-4 offers a larger model better optimized for chat than GPT-3.5-turbo as well as a higher max
    token size (8,192 tokens vs 4,096). GPT-3.5 offers better cost efficiency.

    Note: GPT-4 is currently in limited beta. Make sure that you have access to GPT-4 through
    OpenAI before attempting to use any of the GPT-4 models with OFRAK AI.

    For more information, visit https://platform.openai.com/docs/models/overview.
    """

    THREE_FIVE_TURBO = "gpt-3.5-turbo"
    THREE_FIVE_TURBO_0301 = "gpt-3.5-turbo-0301"
    FOUR = "gpt-4"
    FOUR_0314 = "gpt-4-0314"
    FOUR_32K = "gpt-4-32k"
    FOUR_32K_0314 = "gpt-4-32k-0314"


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
    model: str = ModelType.THREE_FIVE_TURBO
    system_message: Optional[str] = None
    temperature: float = 1.0


async def get_chatgpt_response(
    history: List[Dict[str, str]],
    max_tokens: int,
    config: ChatGPTConfig,
) -> OpenAIObject:
    """
    Calls the OpenAI API with the appropriate model and message history while performing
    exponential backoff in case of rate limit errors.

    :param history: a history of messages conforming to the OpenAI API specification
    :param max_tokens: a maximum number of tokens to include in the model's response before
        truncation occurs
    :param config: an instance of ChatGPTConfig with the desired model parameters to use

    :raises OpenAIError: if unable to make a valid request or receive a response from ChatGPT

    :return: a model response in the form of an OpenAIObject if the call succeeds
    """

    @retry_with_exponential_backoff
    async def retry_response(**kwargs) -> OpenAIObject:
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
