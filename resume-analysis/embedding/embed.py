import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# 데이터 로드
df = pd.read_csv("../data/processed_resume_data.csv")

docs = df["text"].tolist()

print("문서 개수:", len(docs))

# 모델 로드
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# embedding 생성
embeddings = model.encode(
    docs,
    batch_size=64,
    show_progress_bar=True
)

print("embedding shape:", embeddings.shape)

# 저장
np.save("../data/resume_embeddings.npy", embeddings)

print("embedding 저장 완료")