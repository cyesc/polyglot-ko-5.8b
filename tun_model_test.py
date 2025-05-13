from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# ✅ 모델 로딩
base_model_name = "EleutherAI/polyglot-ko-5.8b"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ✅ 튜닝된 LoRA adapter 로드
adapter_path = "./output_v2/checkpoint_0513_v1"
model = PeftModel.from_pretrained(base_model, adapter_path)

# ✅ 인터랙티브 테스트 루프
while True:
    instruction = input("\n📥 질문을 입력하세요 (종료하려면 'exit'): ")
    if instruction.lower() == "exit":
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
