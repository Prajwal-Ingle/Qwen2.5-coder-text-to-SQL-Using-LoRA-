# Text-to-SQL Fine-Tuning with LoRA

**Parameter-Efficient Fine-Tuning of Qwen2.5-Coder-1.5B-Instruct for Natural Language to SQL Generation**

This project fine-tunes [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) to convert natural-language questions and SQL database schemas into executable SQL queries.

Using Low-Rank Adaptation (LoRA), the model is specialized for Text-to-SQL generation without updating its original weights. Training is performed in FP16 precision without 4-bit quantization, using a single NVIDIA Tesla T4 GPU.

The primary objective is to investigate how parameter-efficient fine-tuning improves SQL generation accuracy compared with the original pretrained model.

## Performance Evaluation

The base model and fine-tuned model were evaluated using an in-memory SQLite execution harness.

<escape>| Metric | Base Model | Fine-Tuned Model | Improvement |
|---|---:|---:|---:|
| Execution Accuracy | 81.82% | **90.91%** | **+9.09 pp** |
| Valid SQL Rate | 91.61% | **98.60%** | **+6.99 pp** |
| Clean Format Rate | 100.00% | 100.00% | — |</escape>

<escape>**Evaluation sample: 143 executable examples**</escape>

### Key Findings

- **Improved execution accuracy:** Increased from 81.82% to 90.91%, demonstrating better query generation performance after fine-tuning.
- **Higher SQL validity:** Improved from 91.61% to 98.60%, reducing SQL syntax and execution failures.
- **Consistent output formatting:** Both models maintained a 100% clean-format rate.
- **Lightweight adaptation:** The model was fine-tuned using LoRA adapters rather than updating all 1.54B parameters.

## Dataset

The experiment uses the [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) dataset.

Each example contains:
- Database schema represented using SQL `CREATE TABLE` statements.
- A natural-language question.
- The corresponding ground-truth SQL query.

A reproducible sample of 3,000 examples was selected using random seed 42.

| Dataset Split | Examples |
|---|---:|
| Training | 2,850 |
| Validation | 150 |
| Total | 3,000 |

**Evaluation clarification:** The execution benchmark uses 143 testable examples from the 150-example held-out validation split. Seven examples were not included in the reported execution metrics. The reported accuracy denominators are therefore based on 143 examples, not 150.

The validation examples are kept separate from the training examples for evaluating the fine-tuned model.

## Fine-Tuning Configuration

### Base Model

`Qwen/Qwen2.5-Coder-1.5B-Instruct`

The original model is loaded in FP16 precision and its weights remain frozen during training. LoRA adapters are attached to selected attention projection layers.

### LoRA Configuration

| Parameter | Value |
|---|---|
| Fine-tuning approach | LoRA (PEFT) |
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Trainable base weights | None |
| Quantization | None |

### Training Hyperparameters

| Parameter | Value |
|---|---|
| Framework | Hugging Face TRL SFTTrainer |
| Precision | FP16 |
| Epochs | 1 |
| Batch size per device | 2 |
| Gradient accumulation | 8 |
| Effective batch size | 16 |
| Learning rate | 2e-4 |
| Optimizer | AdamW |
| Scheduler | Cosine |
| Warmup steps | 10 |
| Maximum sequence length | 512 |

### Hardware

- GPU: NVIDIA Tesla T4
- GPU Memory: 16 GB
- Environment: Kaggle Notebook
- Training precision: FP16

## System Architecture

The system accepts a database schema and a natural-language question, formats them using the ChatML conversation template, and passes the prompt to the fine-tuned language model.

```text
       Database Schema
              +
    Natural Language Query
              |
              v
       ChatML Prompt
              |
              v
   Qwen2.5-Coder-1.5B
        Frozen Base
              +
        LoRA Adapters
              |
              v
       SQL Generation
              |
              v
      SQLite Execution
              |
              v
     Result Set Comparison
              |
              v
      Accuracy Evaluation
```

### Evaluation Methodology

The evaluation uses execution-based comparison rather than relying exclusively on exact SQL string matching.

1. The base model generates SQL queries for the held-out examples.
2. The fine-tuned model generates SQL queries for the same examples.
3. Generated queries are evaluated using an in-memory SQLite execution environment.
4. Execution success and SQL validity are recorded.
5. Generated query results are compared against the ground-truth query results.

**Execution accuracy** represents the percentage of evaluated examples for which the generated SQL produces the expected result.

The reported metrics are based on the 143 testable validation examples.

Note: The benchmark is specific to the SQLite evaluation environment. Result-order handling, duplicate-row handling, NULL comparison, and database-specific edge cases should be interpreted according to the implementation in the accompanying notebook.

## Repository Structure

```text
.
├── README.md
└── qwen2.5-coder-1.5b-sql-lora-kaggle.ipynb
```

| File | Description |
|---|---|
| README.md | Project documentation and benchmark results |
| qwen2.5-coder-1.5b-sql-lora-kaggle.ipynb | Complete training, fine-tuning, and evaluation workflow |

## Getting Started

### Requirements

- Kaggle or Google Colab account
- NVIDIA T4 GPU runtime (16 GB VRAM)
- Internet access enabled
- Python environment with the required dependencies

### Running the Notebook

1. Download `qwen2.5-coder-1.5b-sql-lora-kaggle.ipynb`.
2. Upload the notebook to Kaggle or Google Colab.
3. Enable GPU acceleration and select a T4 runtime.
4. Enable internet access to download the model and dataset.
5. Execute the notebook cells sequentially.

The notebook covers:

- Dataset loading and preprocessing.
- ChatML prompt construction.
- Zero-shot evaluation of the original model.
- LoRA adapter configuration.
- Supervised fine-tuning using SFTTrainer.
- Adapter saving.
- SQLite-based execution evaluation.
- Base versus fine-tuned performance comparison.

## Inference

The trained adapter can be loaded alongside the original Qwen2.5-Coder model.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
adapter_dir = "./qwen2.5-coder-1.5b-sql-lora-adapter"

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    adapter_dir
)

model.eval()

messages = [
    {
        "role": "system",
        "content": (
            "You are a SQL assistant. Given a table schema "
            "and a question, reply with ONLY the SQL query."
        )
    },
    {
        "role": "user",
        "content": """Schema:
CREATE TABLE employees (
    id INT,
    name TEXT,
    department TEXT,
    salary INT
);

Question:
List the names of employees in the Engineering department
earning more than 100000."""
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False
    )

sql = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
).strip()

print(sql)
```

The adapter directory must contain the saved LoRA configuration and weights. The original base model is still required for inference.

## Future Improvements

Potential directions for improving the model include:

- **Larger training dataset:** Experiment with 15,000–30,000 training examples to improve coverage of SQL patterns.
- **Extended training:** Compare two and three epochs against the current single-epoch configuration.
- **Expanded LoRA targets:** Include MLP projection layers such as `gate_proj`, `up_proj`, and `down_proj`.
- **Completion-only loss:** Mask instruction and input tokens so that optimization focuses on the target SQL completion.
- **Cross-dialect evaluation:** Test generated queries against PostgreSQL and other SQL engines.
- **Generalization testing:** Evaluate performance on independent Text-to-SQL benchmarks and unseen database schemas.

## Limitations

- The reported benchmark contains 143 testable examples, which is relatively small.
- The evaluation is based on SQLite and does not establish compatibility with every SQL dialect.
- Execution accuracy can vary depending on database contents, query semantics, and result comparison methodology.
- Performance on unseen schemas and complex multi-table queries requires additional validation.
- The results represent this particular dataset, training configuration, and evaluation setup; they should not be interpreted as a universal measure of Text-to-SQL performance.

## References

- [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
- [SQL-Create-Context Dataset](https://huggingface.co/datasets/b-mc2/sql-create-context)
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [Hugging Face TRL](https://github.com/huggingface/trl)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

---

**Focus Areas:** Large Language Models · Text-to-SQL · Parameter-Efficient Fine-Tuning · LoRA · Supervised Fine-Tuning · SQL Execution Evaluation · Python · SQLite
