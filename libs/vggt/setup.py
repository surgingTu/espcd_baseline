from setuptools import setup, find_packages

setup(
    name="vggt",
    version="0.0.1",
    packages=find_packages(),  # 自动找到 vggt/
    install_requires=[
        "torch",
        "torchvision",
    ],
)