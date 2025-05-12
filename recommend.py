from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from model_utils import generate_response

embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def extract_conditions(response):
    theme, age_list = "", []
    for line in response.split('\n'):
        if 'theme:' in line:
            theme = line.split('theme:')[1].strip()
        elif 'age:' in line:
            age_list = line.split('age:')[1].strip().split(',')
    return theme, age_list

def check_age_match(book_age, extracted_ages):
    if pd.isna(book_age):
        return False
    return any(age.strip() in book_age for age in extracted_ages)

def get_recommendations(user_input, books_df, theme, age_list):
    filtered_books = books_df[
        books_df['theme'].str.contains(theme, na=False) &
        books_df['age'].apply(lambda x: check_age_match(x, age_list))
    ]

    if len(filtered_books) == 0:
        return None, None

    book_texts = filtered_books['summary'] + ' ' + filtered_books['tags'] + ' ' + filtered_books['theme']
    book_vectors = embedding_model.encode(book_texts.tolist(), convert_to_numpy=True)
    query_vector = embedding_model.encode([user_input], convert_to_numpy=True)

    index = faiss.IndexFlatL2(book_vectors.shape[1])
    index.add(book_vectors)
    _, topk_indices = index.search(query_vector, k=5)

    top_books = filtered_books.iloc[topk_indices[0]].reset_index(drop=True)
    top_book = top_books.iloc[0]

    reason_prompt = f"""
[질문] {user_input}
[책 정보]
제목: {top_book['title']}
요약: {top_book['summary']}
테마: {top_book['theme']}
대상 연령: {top_book['age']}
[답변] 사용자의 요청에 따라 위 책을 추천하는 이유를 설명해주세요.
"""
    reason = generate_response(reason_prompt)

    return top_books.to_dict(orient="records"), reason
