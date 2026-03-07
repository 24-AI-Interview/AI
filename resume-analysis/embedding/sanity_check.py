import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# 데이터 로드
df = pd.read_csv("../data/processed_resume_data.csv")
docs = df["Answer"].astype(str).tolist()
questions = df["Question"].astype(str).tolist()

# embedding 로드 (Answer embedding)
embeddings = np.load("../data/resume_answer_embeddings.npy")

# query용 모델
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

print("문서 개수:", len(docs))
print("embedding shape:", embeddings.shape)

# 확인할 문서 index
query_idx = 100

print("\n==============================")
print("Query 질문")
print("==============================")
print(questions[query_idx][:500])

# Query embedding 생성
query_embedding = model.encode(
    questions[query_idx],
    convert_to_numpy=True
)

# cosine similarity
similarities = cosine_similarity(
    query_embedding.reshape(1, -1),
    embeddings
)[0]

# 상위 유사 문장
top_k = 5
top_indices = similarities.argsort()[-top_k:][::-1]

print("\n==============================")
print("유사 답변")
print("==============================")

for i in top_indices:
    print(f"\nSimilarity: {similarities[i]:.3f}")
    print(docs[i][:500])