import os
import pytest

from ofrak.core.program import Program
from ofrak.ofrak_context import OFRAKContext
from ofrak.resource import Resource
from ofrak.service.resource_service_i import ResourceFilter
from ofrak_ai.chatgpt import ChatGPTAnalysis

INPUT_FILE_PATH = os.path.join(os.path.dirname(__file__), "assets/hello.out")


@pytest.fixture
async def resource(ofrak_context: OFRAKContext) -> Resource:
    resource = await ofrak_context.create_root_resource_from_file(INPUT_FILE_PATH)
    return resource


async def test_sassy_string_modifier(resource: Resource):
    await resource.unpack_recursively()

    program = await resource.get_only_descendant(
        r_filter=ResourceFilter(include_self=True, tags=[Program])
    )
    assert program.has_tag(Program)
    analysis = await program.analyze(ChatGPTAnalysis)

    assert analysis.description != ""
