from setuptools import setup, find_packages

setup(
    name="vagen",
    version="26.2.5",
    packages=find_packages(),
    install_requires=[
        "gym-sokoban",
        "gymnasium",
        "gymnasium[toy-text]",
        "uvicorn<0.41",
    ],
    python_requires=">=3.10",
    entry_points={
        # vLLM loads these in every spawned subprocess via load_general_plugins().
        # This ensures CambrianQwenForCausalLM is registered in EngineCore_DP* procs.
        "vllm.general_plugins": [
            "cambrian_vllm = vagen.models.cambrian_plugin:register",
        ],
    },
)
