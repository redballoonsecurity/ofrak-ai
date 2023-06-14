import setuptools
import pkg_resources
from setuptools.command.egg_info import egg_info


class egg_info_ex(egg_info):
    """Includes license file into `.egg-info` folder."""

    def run(self):
        # don't duplicate license into `.egg-info` when building a distribution
        if not self.distribution.have_run.get("install", True):
            # `install` command is in progress, copy license
            self.mkpath(self.egg_info)

        egg_info.run(self)


# Should be the same as in build_image.py
def read_requirements(requirements_path):
    with open(requirements_path) as requirements_handle:
        return [
            str(requirement)
            for requirement in pkg_resources.parse_requirements(requirements_handle)
        ]


setuptools.setup(
    name="ofrak_ai",
    version="0.0.1",
    description="OFRAK components, powered by AI",
    packages=setuptools.find_packages(exclude=["ofrak_ai_test"]),
    package_data={
        "ofrak_ai": ["py.typed"],
    },
    install_requires=["ofrak~=3.0.0"] + read_requirements("requirements.txt"),
    extras_require={
        "test": read_requirements("requirements-test.txt"),
    },
    author="Red Balloon Security",
    author_email="ofrak@redballoonsecurity.com",
    url="https://ofrak.com/",
    download_url="https://github.com/redballoonsecurity/ofrak",
    project_urls={
        "Documentation": "https://ofrak.com/docs/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Security",
        "Typing :: Typed",
    ],
    python_requires=">=3.8",
    cmdclass={"egg_info": egg_info_ex},
    entry_points={
        "ofrak.packages": ["ofrak_pkg = ofrak"],
        "console_scripts": ["ofrak = ofrak.__main__:main"],
    },
    include_package_data=True,
)
