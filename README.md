Text-to-SQL Fine-Tuning with LoRA (Qwen2.5-Coder-1.5B-Instruct)Parameter-efficient fine-tuning (PEFT) of Qwen/Qwen2.5-Coder-1.5B-Instruct using pure LoRA in FP16 (no 4-bit quantization) to translate natural-language questions and CREATE TABLE schemas into clean, executable SQL queries.Key ResultsEvaluated on 143 testable held-out examples from b-mc2/sql-create-context using an in-memory SQLite execution harness comparing predicted vs. ground-truth query result sets:MetricBase Model (No LoRA)Fine-Tuned (+LoRA)ImprovementExecution Accuracy81.82% (117/143)90.91% (130/143)+9.09%Valid SQL Rate91.61% (131/143)98.60% (141/143)+6.99%Clean Format Rate100.00%100.00%—Highlights+9.09% jump in Execution Accuracy: Beyond matching formatting conventions, LoRA fine-tuning improved semantic query accuracy over the already strong base code model.98.60% Syntactically Valid SQL: Reduced SQLite execution/syntax errors from ~12 down to just 2 queries across the held-out test set.Lightweight Adapter Footprint: Trained only attention projection adapters (r=16, alpha=32), keeping trainable parameters minimal and fast to train on a single Tesla T4 GPU.Project Architecture & WorkflowSchema (DDL) + Question
         │
         ▼
  ChatML Prompt Template
         │
         ▼
Qwen2.5-Coder-1.5B-Instruct (FP16 Frozen Base)
         +
   LoRA Adapters (r=16, alpha=32 on q,k,v,o_proj)
         │
         ▼
   Generated SQL Query ──► In-Memory SQLite Execution Evaluator
Configuration SummaryBase Model: Qwen/Qwen2.5-Coder-1.5B-Instruct (~1.54B parameters, loaded in torch.float16)Dataset: b-mc2/sql-create-context — 3,000 sampled rows (seed=42), split into 2,850 train and 150 validation examplesLoRA Hyperparameters:Rank (r): 16Scaling (lora_alpha): 32Dropout (lora_dropout): 0.05Target Modules: ["q_proj", "k_proj", "v_proj", "o_proj"]Training Setup (trl.SFTTrainer):Precision: fp16=True, bf16=FalseEpochs: 1Effective Batch Size: 16 (per_device_train_batch_size=2, gradient_accumulation_steps=8)Optimizer & Scheduler: adamw_torch, learning_rate=2e-4, cosine decay with 10 warmup stepsMax Sequence Length: 512Hardware: Kaggle Notebook — NVIDIA Tesla T4 GPU (16 GB VRAM)Repository Contents├── README.md                                   # Project overview, methodology, and benchmarks
└── qwen2.5-coder-1.5b-sql-lora-kaggle.ipynb    # End-to-end training & SQLite evaluation notebook
Quickstart & Usage1. Run the NotebookUpload qwen2.5-coder-1.5b-sql-lora-kaggle.ipynb to Kaggle (enable GPU T4 and Internet in notebook settings) or Google Colab (T4 runtime).Run the cells sequentially to:Load Qwen/Qwen2.5-Coder-1.5B-Instruct in float16Benchmark zero-shot baseline SQL generationFine-tune LoRA adapters with SFTTrainerRun the SQLite execution accuracy comparison (Base vs. +LoRA)2. Inference with Saved LoRA Adaptersimport torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
adapter_dir = "./qwen2.5-coder-1.5b-sql-lora-adapter"

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()

messages = [
    {
        "role": "system",
        "content": "You are a SQL assistant. Given a table schema and a question, reply with ONLY the SQL query, nothing else.",
    },
    {
        "role": "user",
        "content": (
            "Schema:\nCREATE TABLE employees (id INT, name TEXT, department TEXT, salary INT);\n\n"
            "Question: List the names of employees in the Engineering department earning more than 100000."
        ),
    },
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=120, do_sample=False)

sql = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
print(sql)
Future ImprovementsScale Data & Epochs: Train on a larger slice (e.g., 15k–30k examples from sql-create-context) for 2–3 epochs to close the remaining ~9% semantic logic gap.Expand LoRA Target Modules: Include MLP projections (gate_proj, up_proj, down_proj) alongside attention layers.Completion-Only Loss Masking: Mask system and schema prompt tokens during training so gradients optimize purely on target SQL tokens.
