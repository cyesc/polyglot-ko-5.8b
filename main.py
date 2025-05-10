from recommend import extract_conditions, get_recommendations
from model_utils import generate_response
import pandas as pd
import json

with open("datasets/books_merged.json", "r", encoding="utf-8") as f:
    books_df = pd.json_normalize(json.load(f))

print("✅ 추천 시스템 준비 완료!")

while True:
    user_input = input("\n질문을 입력하세요 : ")
    if user_input.lower() == "exit":
        break

    # Step 1. 조건 추출
    instruction_prompt = f"[질문] {user_input}\n[답변] theme: "
    cond_response = generate_response(instruction_prompt)
    theme, age_list = extract_conditions(cond_response)

    # Step 2 ~ 4. 추천 및 이유 생성
    top_books, reason = get_recommendations(user_input, books_df, theme, age_list)

    if top_books is None:
        print("😢 조건에 맞는 도서를 찾을 수 없습니다.")
        continue

    print("\n📚 추천 도서:", top_books[0]['title'])
    print("🧠 추천 이유:", reason)
    print("📌 기타 추천:")
    for book in top_books[1:]:
        print("  -", book['title'])
