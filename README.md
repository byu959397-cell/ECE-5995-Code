# TGR-LoRA: Task Gradient Routing LoRA for Continual Learning

## 1. Data Preparation

To replicate the experiments, you need to prepare the TRACE benchmark dataset. (https://github.com/BeyonderXX/TRACE)

1.  Create a directory named "data/trace" in the root of the project.
2.  Download the datasets and organize them **strictly** according to the structure below.
3.  Ensure every task folder contains exactly three files: "train.json", "eval.json", and "test.json".

**Directory Structure:**

```text
data/
└── trace/
    ├── 20Minuten/
    │   ├── eval.json
    │   ├── test.json
    │   └── train.json
    ├── C-STANCE/
    │   ├── eval.json
    │   ├── test.json
    │   └── train.json
    ├── FOMC/
    │   ├── eval.json
    │   ├── test.json
    │   └── train.json
    ├── MeetingBank/
    │   ├── eval.json
    │   ├── test.json
    │   └── train.json
    ├── NumGLUE-cm/
    │   ├── eval.json
    │   ├── test.json
    │   └── train.json
    ├── NumGLUE-ds/
    │   ├── eval.json
    │   ├── test.json
    │   └── train.json
    ├── Py150/
    │   ├── eval.json
    │   ├── test.json
    │   └── train.json
    └── ScienceQA/
        ├── eval.json
        ├── test.json
        └── train.json


tgr_lora_project/
├── data/
│   └── trace/ ...        
├── outputs/
│   └── ...              
├── tgr_lora/           
│   ├── __init__.py
│   ├── config.py        
│   ├── math_utils.py   
│   ├── memory_bank.py   
│   ├── lora_layers.py 
│   ├── trainer.py      
│   ├── trace_dataset.py 
│   └── evaluation.py    
├── train_trace.py     
├── requirements.txt    
└── README.md            


## 2. Prepare the environment
run "pip install -r requirements.txt", the experiment also need to be run with at least one GPU.

## 3. Run the command as follows, you can refer to train_trace.py file to add other commands 
HF_ENDPOINT=https://hf-mirror.com python train_trace.py \
    --data_path ./data/trace \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --num_tasks 5 \
    --output_dir ./outputs
    
