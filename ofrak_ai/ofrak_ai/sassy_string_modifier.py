import logging
import openai
import string

from dataclasses import dataclass, field
from enum import Enum
from tiktoken import Encoding, encoding_for_model
from typing import Dict, List, Optional

from ofrak import Resource
from ofrak.core.strings import AsciiString, StringPatchingConfig, StringPatchingModifier
from ofrak.component.modifier import Modifier
from ofrak_ai.chatgpt import ChatGPTConfig
from ofrak_ai.exponential_backoff import retry_with_exponential_backoff

LOGGER = logging.getLogger(__name__)


class StringTypeEnum(Enum):
    IDENTIFIER = 0
    SENTENCE = 1


@dataclass
class SassyStringModifierConfig(ChatGPTConfig):
    min_length: int = 50
    encoding: Encoding = encoding_for_model(ChatGPTConfig.model)
    max_retries: int = 3
    prompt_parts: Dict[StringTypeEnum, str] = field(default_factory=dict)


class SassyStringModifier(Modifier[SassyStringModifierConfig]):
    targets = (AsciiString,)

    async def modify(
        self, resource: Resource, config: SassyStringModifierConfig
    ) -> None:
        if not config.prompt_parts:
            config.prompt_parts = {
                StringTypeEnum.IDENTIFIER: "It is EXTREMELY important that your entire response contains no spaces. ",
                StringTypeEnum.SENTENCE: "If the input string contains any C format specifiers, then it is EXTREMELY\
                    important that your response contains the same specifiers in the same order. ",
            }

        string = await resource.view_as(AsciiString)
        text = string.Text
        text_length = len(text)

        if text_length >= config.min_length:
            # Assume strings without spaces must remain space-free
            if " " not in text:
                str_type = StringTypeEnum.IDENTIFIER
            else:
                str_type = StringTypeEnum.SENTENCE
            result = self.get_modified_string(text, text_length, str_type, config)
            if result:
                string_patch_config = StringPatchingConfig(offset=0, string=result)
                await resource.run(StringPatchingModifier, string_patch_config)

    def get_modified_string(
        self,
        text: str,
        text_length: int,
        str_type: StringTypeEnum,
        config: SassyStringModifierConfig,
    ) -> str:
        num_tokens = len(config.encoding.encode(text))

        history = [
            {
                "role": "user",
                "content": f"You are a sassy person. I will send a message and you will respond by making the text of the message more sassy.\
                                The sassy text you generate must be shorter or equal to the length to the length of the original message.\
                                It is EXTREMELY important that your sassy version is shorter than the original and contains only ASCII characters.\
                                {(str_type == StringTypeEnum.IDENTIFIER) * config.prompt_parts.get(StringTypeEnum.IDENTIFIER, '')} \
                                {(str_type == StringTypeEnum.SENTENCE) * config.prompt_parts.get(StringTypeEnum.SENTENCE, '')} \
                                If you understand, make the following message more sassy: \n{text}",
            },
        ]
        # print(history)

        try:
            response = self.get_chatgpt_response(history, num_tokens * 2, config)

            if response:
                retries = 0
                print(f"original text: {text}")
                print(f"chatgpt response: {response.choices[0].message.content}")
                if str_type == StringTypeEnum.IDENTIFIER:
                    # Since ChatGPT likes to add commentary, assume the longest word in the response is the sassified input
                    result = max(response.choices[0].message.content.split(), key=len)
                    print(f"max word: {result}")
                else:
                    result = response.choices[0].message.content
                while len(result) > text_length and retries < config.max_retries:
                    retries += 1
                    history.extend(
                        [
                            {
                                "role": "assistant",
                                "content": response.choices[0].message.content,
                            },
                            {
                                "role": "user",
                                "content": f"Make it shorter.",
                            },
                        ]
                    )
                    try:
                        response = self.get_chatgpt_response(
                            history, text_length * 2, config
                        )

                        print(f"original text: {text}")
                        print(
                            f"chatgpt response: {response.choices[0].message.content}"
                        )
                        if str_type == StringTypeEnum.IDENTIFIER:
                            result = max(
                                response.choices[0].message.content.split(), key=len
                            )
                            print(f"max word: {result}")
                    except Exception as e:
                        LOGGER.warning(f"Exception {e} occurred, skipped {text}")

            # ChatGPT will sometimes add non-ASCII characters like emojis even when asked not to
            result = self.remove_unicode(result)
            return result[: text_length - 1]

        except Exception as e:
            LOGGER.warning(f"Exception {e} occurred, skipped {text}")

    def get_chatgpt_response(
        self, history: List[str], max_tokens: int, config: SassyStringModifierConfig
    ) -> Optional[str]:
        @retry_with_exponential_backoff
        def retry_response(**kwargs) -> Optional[str]:
            return openai.ChatCompletion.create(**kwargs)

        return retry_response(
            model=config.model,
            temperature=config.temperature,
            max_tokens=max_tokens,
            messages=[message for message in history],
        )

    def remove_unicode(self, text: str) -> str:
        printable = set(string.printable)
        return "".join(filter(lambda x: x in printable, text))
