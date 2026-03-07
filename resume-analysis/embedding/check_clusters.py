import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt

# 데이터 로드
df = pd.read_csv("../data/processed_resume_data.csv")
docs = df["Answer"].astype(str).tolist()

# embedding 로드
embeddings = np.load("../data/resume_answer_embeddings.npy")

print("문서 수:", len(docs))
print("embedding shape:", embeddings.shape)

# ==========================
# 1. 차원 축소 (UMAP)
# ==========================

reducer = umap.UMAP(
    n_neighbors=15,
    n_components=2,
    metric="cosine",
    random_state=42
)

emb_2d = reducer.fit_transform(embeddings)

# ==========================
# 2. 클러스터링
# ==========================

k = 8  # 보통 자소서 데이터는 6~10개 정도 cluster 나옴

kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(embeddings)

df["cluster"] = labels

# ==========================
# 3. 클러스터 시각화
# ==========================

plt.figure(figsize=(8,6))
plt.scatter(
    emb_2d[:,0],
    emb_2d[:,1],
    c=labels,
    cmap="tab10",
    s=5
)
plt.title("Resume Embedding Clusters")
plt.show()

# ==========================
# 4. 각 클러스터 예시 출력
# ==========================

print("\n===== Cluster Examples =====")

for c in range(k):

    print(f"\nCluster {c}")
    print("-"*40)

    examples = df[df["cluster"] == c]["Answer"].head(3)

    for e in examples:
        print(e[:200])
        print()