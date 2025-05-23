from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch
import json

# 1. 데이터 로드 함수
def load_dataset_from_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    inputs = [ex["instruction"] + "\n\n" + ex["input"] for ex in data]
    outputs = [ex["output"] for ex in data]
    return Dataset.from_dict({"input": inputs, "output": outputs})

# 2. 토크나이즈 함수
def tokenize(example):
    prompt = f"{example['input']}\n\n응답: {example['output']}"
    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# 3. 모델 불러오기
model_name = "EleutherAI/polyglot-ko-5.8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 4. LoRA 설정
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, peft_config)

# 5. 데이터 로딩 및 전처리
dataset = load_dataset_from_jsonl("datasets/instruction_dataset_high_precision.jsonl")
tokenized_dataset = dataset.map(tokenize)

# 6. 학습 설정
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    evaluation_strategy="no"
)

# 7. Trainer 구성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

# 8. 학습 실행
trainer.train()
