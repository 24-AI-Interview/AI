import pandas as pd
import numpy as np
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

# 데이터 로드
df = pd.read_csv("../data/processed_resume_data.csv")
docs = df["text"].tolist()

# embedding 로드
embeddings = np.load("../data/resume_embeddings.npy")

print("문서 개수:", len(docs))
print("embedding shape:", embeddings.shape)

# UMAP 설정
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)

# HDBSCAN 설정
hdbscan_model = HDBSCAN(
    min_cluster_size=30,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

# BERTopic 모델 생성
topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=True,
    verbose=True
)

# 학습
topics, probs = topic_model.fit_transform(docs, embeddings)

# 토픽 정보
topic_info = topic_model.get_topic_info()

# 노이즈 계산
noise_row = topic_info[topic_info.Topic == -1]
noise_count = noise_row["Count"].iloc[0] if len(noise_row) > 0 else 0
noise_ratio = noise_count / len(docs)

print("\n==============================")
print("모델 결과")
print("==============================")
print("토픽 개수:", len(topic_info) - 1)
print("노이즈 문서:", noise_count)
print("노이즈 비율:", round(noise_ratio, 3))

# 토픽 정보 저장
topic_info.to_csv("../data/topic_info_final.csv", index=False)
# 모델 저장
topic_model.save("../data/bertopic_model_final")

print("\nBERTopic 모델 저장 완료")
