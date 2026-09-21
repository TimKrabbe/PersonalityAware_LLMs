# PersonalityAware_LLMs
##### Short description:
A repository containing an LLM fine-tuning and evaluation framework to create a personality-aware, LLM-based dialogue engine for social simulations.

## Personality-Aware LLMs
This repository contains the code for the paper "Do Personality-Tuned LLMs Make Better Social Agents?", submitted to IEEE ICRA 2027.

The project investigates whether fine-tuning small language models on personality-labelled text can improve the representation of personality in LLM-based social simulations. We use the Myers-Briggs Type Indicator (MBTI) as a framework for modelling personality and evaluate generated dialogues across different social scenarios.

### Repository Structure
preprocessing/ – Data preprocessing and dataset preparation
model_training/ – Model fine-tuning and training scripts
dialogue_generation/ – Generation of personality-conditioned dialogues
evaluation/ – Quantitative and qualitative evaluation

### Models
The experiments use the following base models:

Qwen2.5-7B-Instruct
Ministral-8B-Instruct-2410

Both base models and LoRA fine-tuned variants are evaluated.

### Reproducibility
The code and configuration files required to reproduce the experiments are provided in this repository. The experiments require GPU resources and may take substantial computational time.

The datasets used in the study are available separately on Hugging Face where permitted by the respective data and licensing conditions.

### License
See the repository and individual dataset/model licenses for applicable licensing information.