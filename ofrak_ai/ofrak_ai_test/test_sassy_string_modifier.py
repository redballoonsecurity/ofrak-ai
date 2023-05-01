import asyncio
import os
import pytest
import subprocess

from ofrak.ofrak_context import OFRAKContext
from ofrak.resource import Resource
from ofrak.service.resource_service_i import ResourceFilter
from ofrak.core.strings import AsciiString
from ofrak_ai.sassy_string_modifier import (
    SassyStringModifier,
    SassyStringModifierConfig,
)

SOURCE_DIR = os.path.join(os.path.dirname(__file__), "assets/")
SOURCE_FILE = "regular_strings.c"
INPUT_FILE_NAME = "regular_strings"
OUTPUT_FILE_NAME = "sassy_strings"


@pytest.fixture
def elf_file(tmp_path):
    source_path = os.path.join(tmp_path, SOURCE_FILE)
    with open(os.path.join(SOURCE_DIR, SOURCE_FILE)) as f_in, open(
        source_path, "w"
    ) as f_out:
        for line in f_in:
            f_out.write(line)

    regular_path = os.path.join(tmp_path, INPUT_FILE_NAME)
    cmd = ["gcc", "-o", regular_path, source_path]
    proc = subprocess.run(cmd)
    return regular_path


@pytest.fixture
async def resource(ofrak_context: OFRAKContext, elf_file) -> Resource:
    resource = await ofrak_context.create_root_resource_from_file(elf_file)
    return resource


async def test_sassy_string_modifier(resource: Resource, elf_file):
    original = subprocess.run(elf_file, capture_output=True, text=True)
    await resource.unpack_recursively()

    child_strings = list(
        await resource.get_descendants(r_filter=ResourceFilter(tags=[AsciiString]))
    )

    tasks = list()
    for string in child_strings:
        tasks.append(string.run(SassyStringModifier, SassyStringModifierConfig()))
    await asyncio.gather(*tasks)

    sassy_path = os.path.join(os.path.dirname(elf_file), OUTPUT_FILE_NAME)
    await resource.flush_to_disk(sassy_path)

    subprocess.run(["chmod", "+x", sassy_path])
    modified = subprocess.run(sassy_path, capture_output=True, text=True)
    assert modified.returncode == original.returncode
    assert original.stdout != modified.stdout
