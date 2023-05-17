import logging

from dataclasses import dataclass
from openai.error import OpenAIError
from tiktoken import Encoding, encoding_for_model
from typing import List

from ofrak import Resource, ResourceFilter
from ofrak.component.analyzer import Analyzer
from ofrak.core.complex_block import ComplexBlock
from ofrak.core.program import Program
from ofrak.core.strings import AsciiString
from ofrak_ai.chatgpt import ChatGPTAnalysis, ChatGPTConfig, get_chatgpt_response

LOGGER = logging.getLogger(__name__)


@dataclass
class ChatGPTProgramAnalyzerConfig(ChatGPTConfig):
    """
    :param min_length: minimum length string to pull from target Program to be sent to ChatGPT
        for analysis - useful for meeting a model's token constraints
    """

    encoding: Encoding = encoding_for_model(ChatGPTConfig.model)
    min_length: int = 20


class ChatGPTProgramAnalyzer(Analyzer[ChatGPTProgramAnalyzerConfig, ChatGPTAnalysis]):
    # targets = (Program,)
    outputs = (ChatGPTAnalysis,)

    async def analyze(
        self, resource: Resource, config: ChatGPTProgramAnalyzerConfig
    ) -> ChatGPTAnalysis:
        if not config:
            config = ChatGPTProgramAnalyzerConfig()

        program = await resource.view_as(Program)
        code_regions = await program.get_code_regions()
        cbs = []
        for code_region in code_regions:
            cbs.extend(
                list(
                    await code_region.resource.get_descendants_as_view(
                        v_type=ComplexBlock,
                        r_filter=ResourceFilter(tags=(ComplexBlock,)),
                    )
                )
            )
        names = []
        for cb in cbs:
            names.append(cb.Symbol)

        string_resources = await resource.get_descendants_as_view(
            AsciiString, r_filter=ResourceFilter(tags=(AsciiString,))
        )

        strings = [string.Text for string in string_resources]

        responses = []
        batches = self.batch_request_text(strings, config)
        responses.extend(await self.get_batch_responses(batches, config, "strings"))

        batches = self.batch_request_text(names, config, ignore_min=True)
        responses.extend(
            await self.get_batch_responses(batches, config, "names of symbols")
        )

        return ChatGPTAnalysis("\n".join(responses))

    def batch_request_text(self, texts: List[str], config, ignore_min=False):
        batches = []
        curr_batch = []
        token_count = 0
        for text in texts:
            if ignore_min or len(text) > config.min_length:
                num_tokens = len(config.encoding.encode(text))
                # Start new batch once token limit exceeded
                if token_count + num_tokens > 3000:
                    batches.append(curr_batch)
                    curr_batch = []
                    token_count = 0
                curr_batch.append(text)
                token_count += num_tokens

        return batches

    async def get_batch_responses(self, batches, config, prompt):
        responses = []
        for batch in batches:
            history = [
                (
                    {
                        "role": "user",
                        "content": f"Here are {prompt} found in the binary:\n{batch}\n\n\
                                Based on these, what is everything you can tell me about this program? \
                                Explain your reasoning as much as possible.",
                    }
                )
            ]
            print(history)
            try:
                response = await get_chatgpt_response(
                    history=history, max_tokens=400, config=config
                )
                print(response.choices[0].message.content)
                responses.append(response.choices[0].message.content)

            except OpenAIError as e:
                # openai's error messages are rather unhelpful. Log traceback for additional details
                LOGGER.exception("Exception occurred")

        return responses
