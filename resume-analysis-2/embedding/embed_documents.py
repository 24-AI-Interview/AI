# embedding/embed_documents.py
import pandas as pd
import os
import torch
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = 'data/processed/'
FILE_PATH = os.path.join(PROCESSED_DIR, 'documents_prepared.csv')

def main():
    if not os.path.exists(FILE_PATH):
        print("준비된 데이터 파일이 없습니다. prepare_documents.py를 먼저 실행하세요.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"디바이스 확인: {device}")

    # 검증된 SBERT 모델 로딩
    model_name = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'
    print(f"[{model_name}] 모델 로딩 중...")
    model = SentenceTransformer(model_name, device=device)

    # 정제된 데이터 불러오기
    df = pd.read_csv(FILE_PATH)
    df['Document'] = df['Document'].fillna('')
    
    print(f"총 {len(df)}건의 Document 임베딩 시작...")
    
    # 문장 전체를 벡터 공간으로 변환
    embeddings = model.encode(df['Document'].tolist(), show_progress_bar=True)
    
    # 결과 저장
    df['Embedding'] = list(embeddings)
    save_path = os.path.join(PROCESSED_DIR, 'documents_embedded.pkl')
    df.to_pickle(save_path)
    
    print(f"✅ 임베딩 완료 및 저장: {save_path}")

if __name__ == "__main__":
    main()