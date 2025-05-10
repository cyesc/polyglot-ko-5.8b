from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# ✅ 1. 기본 모델과 tokenizer 로드
base_model_name = "EleutherAI/polyglot-ko-5.8b"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# ✅ 2. base model + 튜닝된 adapter 로드
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32,  # NaN 방지용
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./output_v2/checkpoint_2787")  # ← 정확한 경로

# ✅ 3. 실시간 질의 루프
while True:
    instruction = input("\n📥 질문을 입력하세요 (종료하려면 'exit'): ")
    if instruction.strip().lower() == "exit":
        break

    prompt = f"[질문] {instruction}\n[답변]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    print("\n🧠 모델 응답:")
    print(tokenizer.decode(output[0], skip_special_tokens=True).strip())
