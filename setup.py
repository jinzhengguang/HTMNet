from setuptools import setup, find_packages

setup(
    name="lidar_det",
    version="1.0",
    author="Jinzheng Guang",
    author_email="guangjinzheng@qq.com",
    packages=find_packages(
        include=["lidar_det", "lidar_det.*", "lidar_det.*.*"]
    ),
    license="LICENSE.txt",
    description="HTMNet: A Hybrid Transformer-Mamba Networks for LiDAR-based 3D Detection and Semantic Segmentation.",
)
