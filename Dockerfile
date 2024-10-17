# syntax=docker/dockerfile:1
ARG BASE_CONTAINER=ghcr.io/synerbi/sirf:edge-gpu
FROM ${BASE_CONTAINER} AS base

RUN conda install -y monai tensorboard tensorboardx jupytext cudatoolkit=11.8
# monai installs pytorch (CPU), so remove it
RUN pip uninstall -y torch
# last to support cu118
RUN pip install tensorflow[and-cuda]==2.14  
RUN pip install torch --index-url https://download.pytorch.org/whl/cu118
RUN pip install git+https://github.com/TomographicImaging/Hackathon-000-Stochastic-QualityMetrics
RUN pip install pysnooper
