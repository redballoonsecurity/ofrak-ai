import logging
import openai
import string

from dataclasses import dataclass, field
from enum import Enum
from openai.error import OpenAIError
from tiktoken import Encoding, encoding_for_model
from typing import Dict, List, Optional

from ofrak import Resource
from ofrak.core.strings import AsciiString, StringPatchingConfig, StringPatchingModifier
from ofrak.component.modifier import Modifier
from ofrak_ai.chatgpt import ChatGPTConfig, get_chatgpt_response

LOGGER = logging.getLogger(__name__)

SPECIFIER_TYPES = set("diuoxXfFeEgGaAcCsSpn")


class StringType(Enum):
    IDENTIFIER = 0
    SENTENCE = 1


@dataclass
class SassyStringModifierConfig(ChatGPTConfig):
    """
    :param min_length: the minimum string length required for targeting strings
    :param encoding: the tiktoken encoding to use for calculating the number of tokens in a string
    :param max_retries: the maximum number of attempts to ask ChatGPT to meet the prompt specs
        before forcefully truncating the response
    :param prompt_parts: adjustable prompt specifications to give to ChatGPT based on the string
        type
    """

    min_length: int = 50
    encoding: Encoding = encoding_for_model(ChatGPTConfig.model)
    max_retries: int = 3
    prompt_parts: Dict[StringType, str] = field(default_factory=dict)


class SassyStringModifier(Modifier[SassyStringModifierConfig]):
    """
    Targets all [AsciiStrings][ofrak.core.strings.AsciiString] over a specified length, requests
    ChatGPT to sassify them, and patches the sassified strings back into the binary.
    """

    targets = (AsciiString,)

    async def modify(
        self,
        resource: Resource,
        config: SassyStringModifierConfig = SassyStringModifierConfig(),
    ):
        """
        :param resource: the string resource to modify
        """
        # This is technically redundant at the moment since openai does the same thing, but a safe-
        # guard in case openai changes
        openai.api_key = config.api_key
        openai.organization = config.api_organization
        if not config.prompt_parts:
            config.prompt_parts = {
                StringType.IDENTIFIER: "It is EXTREMELY important that your entire response contains no spaces. ",
                StringType.SENTENCE: "If the input string contains any C format specifiers, then it is EXTREMELY\
                    important that your response contains the same specifiers in the same order. ",
            }

        string = await resource.view_as(AsciiString)
        text = string.Text
        text_length = len(text)

        if text_length >= config.min_length:
            # Assume strings without spaces must remain space-free
            if " " not in text:
                str_type = StringType.IDENTIFIER
            else:
                str_type = StringType.SENTENCE
            result = await self._get_modified_string(
                text, text_length, str_type, config
            )
            if result:
                LOGGER.debug(f"Original String: {text}\nSassified String: {result}")
                string_patch_config = StringPatchingConfig(
                    offset=0, string=result, null_terminate=True
                )
                await resource.run(StringPatchingModifier, string_patch_config)

    async def _get_modified_string(
        self,
        text: str,
        text_length: int,
        str_type: StringType,
        config: SassyStringModifierConfig,
    ) -> Optional[str]:
        # Use the number of tokens in the string as an early bounds for response length, under the
        # assumption that we should allow ChatGPT more room for creative responses early in the
        # process and then more forcefully restrict its length after the initial request
        num_tokens = len(config.encoding.encode(text))

        history = [
            {
                "role": "user",
                "content": f"You are a sassy person. I will send a message and you will respond by making the text of the message more sassy.\
                                The sassy text you generate must be shorter or equal to the length to the length of the original message.\
                                It is EXTREMELY important that your sassy version is shorter than the original and contains only ASCII characters.\
                                {(str_type == StringType.IDENTIFIER) * config.prompt_parts.get(StringType.IDENTIFIER, '')} \
                                {(str_type == StringType.SENTENCE) * config.prompt_parts.get(StringType.SENTENCE, '')} \
                                If you understand, make the following message more sassy: \n{text}",
            },
        ]

        try:
            response = await get_chatgpt_response(history, num_tokens * 2, config)

            # Sometimes saw cases where ChatGPT sent no response, so validate there was a response
            if response:
                retries = 0
                # Handle identifier and sentence lengths the same way
                if str_type == StringType.IDENTIFIER:
                    # Since ChatGPT likes to add commentary, assume the longest word in the response
                    # is the sassified input
                    result = max(response.choices[0].message.content.split(), key=len)
                else:
                    result = response.choices[0].message.content
                valid_specifiers = self._verify_specifiers(text, result)
                while (
                    (len(result) > text_length or not valid_specifiers)
                    and retries <= config.max_retries
                    and response
                ):
                    retries += 1
                    history.append(
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    )
                    if not valid_specifiers:
                        history.append(
                            {
                                "role": "user",
                                "content": "Use the same format specifiers in the same order as the original.",
                            }
                        )
                    else:
                        history.append(
                            {
                                "role": "user",
                                "content": "Make it shorter.",
                            }
                        )
                    try:
                        # max_tokens will truncate the generated response before sending it back to
                        # us, so give it a bit more leeway by setting max_tokens = text_length * 2
                        response = await get_chatgpt_response(
                            history, text_length * 2, config
                        )

                        if str_type == StringType.IDENTIFIER and response:
                            result = max(
                                response.choices[0].message.content.split(), key=len
                            )
                        elif response:
                            result = response.choices[0].message.content

                        valid_specifiers = self._verify_specifiers(text, result)

                    except OpenAIError as e:
                        raise e

            # No response with valid specifiers after all retries
            if not self._verify_specifiers(text, result):
                LOGGER.warning(f"Unable to request valid specifiers for ")
                return None
            # ChatGPT will sometimes add non-ASCII characters like emojis even when asked not to
            result = self._remove_unicode(result)
            # Forcefully truncate response if it's still over the length req after all retries
            return result[: text_length - 1]

        except OpenAIError as e:
            # openai's error messages are rather unhelpful. Log traceback for additional details
            LOGGER.exception(f'Exception occurred, skipped "{text}"')

        return None

    def _remove_unicode(self, text: str) -> str:
        printable = set(string.printable)
        return "".join(filter(lambda x: x in printable, text))

    def _verify_specifiers(self, input_text: str, output_text: str) -> bool:
        # Assumes original string has valid specifiers
        input_specifiers = self._extract_specifiers(input_text)
        try:
            output_specifiers = self._extract_specifiers(output_text)
        except ValueError:
            return False

        return input_specifiers == output_specifiers

    def _extract_specifiers(self, text: str) -> List[str]:
        """
        :raises ValueError: if no matching specifier found for a '%' symbol
        """
        results: List[str] = []
        length = len(text)

        # For each format specifier, find the closest following specifier in the string
        for idx, char in enumerate(text):
            # Skip backward-looking escaped '%' signs.
            # TODO: This doesn't handle '%%%' correctly.
            if char == "%" and (idx == 0 or text[idx - 1] != "%"):
                # Skip forward-looking escaped '%' signs
                if idx < length - 1 and text[idx + 1] == "%":
                    continue
                specifier_orders = filter(
                    lambda x: x >= 0,
                    [text.find(specifier, idx) for specifier in SPECIFIER_TYPES],
                )
                closest_match = min(specifier_orders)
                # Capture all optional format arguments
                results.append(text[idx : closest_match + 1])

        return results
