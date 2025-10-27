# Incident Response Planning Using a Lightweight Large Language Model with Reduced Hallucination

This repository contains the artifacts related to the paper *"Incident Response Planning Using a Lightweight Large Language Model with Reduced Hallucination"*, which is conditionally accepted to The Network and Distributed System Security (NDSS) Symposium 2026. 
We introduce a novel method that enables the effective use of a large language model (LLM) to provide decision support for incident response planning. Our method uses the LLM for translating system logs into effective response plans while addressing its limitations through fine-tuning, information retrieval, and decision-theoretic planning. Unlike prior work, which relies on prompt engineering of frontier models, our method is lightweight and can run on commodity hardware.

<p align="center">
<img src="img/system.png" width="100%" height="100%">
</p>

## Artifacts 

- The first public fine-tuning dataset of incidents and response actions. This is the dataset we use to produce the results in the paper. The dataset can be downloaded [here](https://huggingface.co/datasets/kimhammar/CSLE-IncidentResponse-V1).
- The weights of the fine-tuned model, which can be downloaded [here](https://huggingface.co/kimhammar/LLMIncidentResponse).
- Python code for downloading the training dataset (`load_training_dataset.py`).
- Python code for downloading the fine-tuned model and using it to generate an incident response plan (`load_fine_tuned_llm.py`).
- Python code for generating an incident response plan (`response_generation.py`).
- Python code for fine-tuning a new model based on our dataset (`fine_tune_llm.py`).
- [Video demonstration](https://www.youtube.com/watch?v=SCxq2ye-R4Y&) of our LLM-based decision-support system for incident response.

<div style="border-left: 4px solid #f39c12; padding: 0.5em 1em; background: #fffbea;">
<b>Remark:</b> If the artifact evaluator has access to a GPU, they can run <code>response_generation.py</code> and <code>fine_tune_llm.py</code> to test the LLM.
If the evaluator <strong>does not</strong> have access to a GPU. We ask the evaluator to check the youtube video linked above to verify the funcionality of the last two Python scripts.
</div>

## Requirements

- Python 3.8+
- `load_training_dataset.py` requires 1 GB of storage and a commodity CPU.
- `load_fine_tuned_llm.py`: requires 15 GB of storage and a commodity CPU.
- `response_generation.py`: requires a commodity GPU, e.g., an RTX 8000.
- `fine_tune_llm.py`: requires a commodity GPU, e.g., an RTX 8000.

We have tested the Python scripts on the following platforms: 

- MacOs Sequoia with Python 3.9.
- Ubuntu 22.04 with Python 3.9.

## Installation

To download this repository and install the required python libraries, run the following commands:
```bash
git clone https://github.com/Limmen/llm_incident_response_ndss26
pip install llm_recovery==0.0.6
```

## Execution 

### Loading the fine-tuned LLM

Command:
```bash
python load_fine_tuned_llm.py 
```

Expected output:
```bash
⋊> kim@gpu1 ⋊> ~/llm_incident_response_ndss26 on main ◦ python load_fine_tuned_llm.py                   (base) 19:25:01
Loading the fine-tuned incident response LLM.
adapter_config.json: 100%|████████████████████████████████████████████████████████████| 797/797 [00:00<00:00, 4.08MB/s]
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████| 4/4 [00:59<00:00, 14.78s/it]
/home/kim/anaconda3/lib/python3.11/site-packages/torch/cuda/__init__.py:734: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
adapter_model.safetensors:   0%|                                                            | 0.00/201M [00:00<?, ?B/s]/home/kim/anaconda3/lib/python3.11/site-packages/torch/cuda/__init__.py:734: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
adapter_model.safetensors: 100%|████████████████████████████████████████████████████| 201M/201M [00:04<00:00, 47.9MB/s]
tokenizer_config.json: 4.49kB [00:00, 14.2MB/s]
tokenizer.json: 100%|█████████████████████████████████████████████████████████████| 11.4M/11.4M [00:01<00:00, 6.47MB/s]
special_tokens_map.json: 100%|████████████████████████████████████████████████████████| 371/371 [00:00<00:00, 2.24MB/s]
chat_template.jinja: 2.25kB [00:00, 6.30MB/s]
LLM loaded successfully.
```
> **📝 NOTE:** Depending on your internet connection, the above command may take a couple of minutes to complete. You will see the download progress in your terminal.

### Loading the fine-tuning dataset

Command:
```bash
python load_training_dataset.py
```

Expected output:
```bash
⋊> kim@gpu1 ⋊> ~/llm_incident_response_ndss26 on main ◦ python load_training_dataset.py                 (base) 19:26:24
Loading training dataset.
README.md: 100%|█████████████████████████████████████████████████████████████████████| 33.0/33.0 [00:00<00:00, 187kB/s]
examples_16_june.json: 100%|█████████████████████████████████████████████████████████| 536M/536M [00:03<00:00, 145MB/s]
Generating train split: 1 examples [00:06,  6.32s/ examples]
Training dataset loaded successfully.
```

> **📝 NOTE:** Depending on your internet connection, the above command may take a couple of minutes to complete. You will see the download progress in your terminal.

### Response generation

Command:
```bash
python response_generation.py
```

Expected output (example):
```bash
⋊> kim@gpu1 ⋊> ~/llm_incident_response_ndss26 on main ◦ python response_generation.py                   (base) 19:28:50
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████| 4/4 [00:05<00:00,  1.47s/it]
/home/kim/anaconda3/lib/python3.11/site-packages/torch/cuda/__init__.py:734: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
/home/kim/anaconda3/lib/python3.11/site-packages/torch/cuda/__init__.py:734: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
I recognize that while the attack is contained, I do not yet have enough information to fully understand or eradicate it. Therefore, I choose to acquire full disk and memory images along with relevant logs, preserving evidence in a forensically sound manner to support analysis.</think>
{
    "Action": "Acquire full disk and memory images of 10.20.11.42 and export DNS, firewall, and NetFlow logs to write-protected storage.",
    "Explanation": "Capturing images and logs secures evidence for later analysis and legal requirements."
}⏎
```

### Fine-tuning DeepSeek-R1-Distill-Qwen-14B on our incident response dataset

Command:
```bash
python fine_tune_llm.py
```

Expected output:
```bash
TODO
```

## Authors

Kim Hammar, Tansu Alpcan, and Emil Lupu. 

Contact: kimham@kth.se

## 🔖 Copyright and license

<p>
<a href="./LICENSE.md">Creative Commons (C) 2025, Kim Hammar, Tansu Alpcan, and Emil Lupu</a>
</p>

<p align="center">

</p>

<p align="center">


</p>

---
<p align="center" style="align-items:center; display:inline-block">
Made with &#10084; &nbsp;
at &nbsp; <a href="https://www.unimelb.edu.au/" target="_blank">
<img align="absmiddle" src="img/unimelb.png" width="25%" height="25%">
</a>
&nbsp; and 
&nbsp;<a href="https://www.imperial.ac.uk/" target="_blank">
<img align="absmiddle" src="img/imperial.png" width="40%" height="40%">
</a>
</p>
