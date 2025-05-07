from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "EleutherAI/polyglot-ko-5.8b"

print("🔧 모델 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

print("✅ 모델 로드 완료!")

# 테스트 프롬프트
instruction = "3세 유아에게 추천할 수 있는 책을 알려줘"
inputs = tokenizer(instruction, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.95,
        do_sample=True  # 이거 안 넣으면 경고 뜸 (sampling 관련)
    )

print("🧠 모델 응답:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
