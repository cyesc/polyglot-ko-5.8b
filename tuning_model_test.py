from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 모델 경로 설정
base_model_name = "EleutherAI/polyglot-ko-5.8b"
adapter_path = "output/checkpoint-2787"

# 모델 로드
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

print("✅ 튜닝된 모델 로드 완료!")

# 터미널 인터랙션
while True:
    user_input = input("\n질문을 입력하세요 : ")
    if user_input.lower().strip() == "exit":
        break

    prompt = f"[질문] {user_input}\n[답변]"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    print("\n🧠 모델 응답:\n", response)
