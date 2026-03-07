import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer


def clean_text(text):

    text = text.lower()

    # OOO 같은 마스킹 제거
    text = re.sub(r"\bo+\b", " ", text)

    # byte 제거
    text = re.sub(r"\d+byte", " ", text)

    # 숫자 제거
    text = re.sub(r"\b\d+\b", " ", text)

    # 특수문자 제거
    text = re.sub(r"[^a-zA-Z가-힣 ]", " ", text)

    # 공백 정리
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==============================
# 데이터 로드
# ==============================

df = pd.read_csv("../data/processed_resume_data.csv")

docs = df["Answer"].astype(str).apply(clean_text).tolist()

print("문서 개수:", len(docs))


# ==============================
# 모델 로드
# ==============================

model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")


# ==============================
# embedding 생성
# ==============================

embeddings = model.encode(
    docs,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("embedding shape:", embeddings.shape)


# ==============================
# 저장
# ==============================

np.save("../data/resume_answer_embeddings.npy", embeddings)

print("embedding 저장 완료")