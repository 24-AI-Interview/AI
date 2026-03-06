import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 데이터 로드
df = pd.read_csv("../data/processed_resume_data.csv")
docs = df["text"].tolist()

# embedding 로드
embeddings = np.load("../data/resume_embeddings.npy")

print("문서 개수:", len(docs))
print("embedding shape:", embeddings.shape)

# 확인할 문서 index
query_idx = 100

print("\n==============================")
print("Query 문장")
print("==============================")
print(docs[query_idx][:500])

# cosine similarity
similarities = cosine_similarity(
    embeddings[query_idx].reshape(1, -1),
    embeddings
)[0]

# 상위 유사 문장
top_k = 5
top_indices = similarities.argsort()[-(top_k+1):-1][::-1]

print("\n==============================")
print("유사 문장")
print("==============================")

for i in top_indices:
    print(f"\nSimilarity: {similarities[i]:.3f}")
    print(docs[i][:500])