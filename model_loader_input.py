from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "EleutherAI/polyglot-ko-5.8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

print("✅ 모델 로드 완료!")

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
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    print("\n🧠 모델 응답:")
    print(tokenizer.decode(output[0], skip_special_tokens=True).strip())
