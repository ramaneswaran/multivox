# MultiVox

[📄 Paper](https://aclanthology.org/2025.emnlp-main.1447/) | [🤗 Dataset](https://huggingface.co/datasets/gamma-lab-umd/multivox)

MultiVox is a benchmark to assess how well omni-modal language models can integrate audio and visual cues to give a contextual repsonse

### Example baseline

We provide scripts to run Qwen 2.5 Omni using vLLM here

```python
python3 src/baseline_qwen.py
```

### Evaluation

We use GPT 4.1-mini to run evaluation. You can use the following script to run evaluation

```python
python3 src/evaluate.py
```
