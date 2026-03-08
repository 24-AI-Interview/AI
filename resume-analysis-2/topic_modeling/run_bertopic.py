import pandas as pd
import os
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from kiwipiepy import Kiwi
from tqdm import tqdm

# 1. 경로 설정
PROCESSED_DIR = 'data/processed/'
FILE_PATH = os.path.join(PROCESSED_DIR, 'documents_embedded.pkl')

# 2. Kiwi 초기화
kiwi = Kiwi()

def get_keywords(text):
    """문서 단위로 명사, 동사, 형용사만 추출해서 띄어쓰기로 연결된 문자열로 반환"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    tokens = kiwi.tokenize(text)
    result = []
    for t in tokens:
        # 명사, 동사, 형용사만 추출하고 1글자는 제외
        if t.tag in ['NNG', 'NNP', 'VV', 'VA'] and len(t.form) > 1:
            result.append(t.form + ('다' if t.tag in ['VV', 'VA'] else ''))
            
    return " ".join(result)

def main():
    if not os.path.exists(FILE_PATH):
        print("임베딩된 데이터가 없습니다.")
        return

    print("데이터 로딩 중...")
    df = pd.read_pickle(FILE_PATH)
    embeddings = np.stack(df['Embedding'].values)
    
    print("문서 사전 토큰화 진행 중 (형태소 분석기로 핵심 단어만 솎아내기)...")
    # tqdm을 사용해 진행률 표시
    tqdm.pandas()
    df['Tokenized_Doc'] = df['Document'].progress_apply(get_keywords)
    
    # 혹시나 토큰화 후 텅 빈 문서가 생기면 에러 방지용으로 채워줌
    df['Tokenized_Doc'] = df['Tokenized_Doc'].replace("", "데이터없음")
    docs = df['Tokenized_Doc'].tolist()
    
    # 3. 이제 CountVectorizer는 복잡한 분석 없이 '띄어쓰기' 기준으로만 단어를 자르면 됨!
    vectorizer_model = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
    
    print("\n🚀 BERTopic 모델 학습 시작 (이제 에러 안 날 거야!)...")
    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        verbose=True
    )
    
    topics, probs = topic_model.fit_transform(docs, embeddings)
    df['Topic'] = topics
    
    print("\n=== 🌟 추출된 상위 10개 핵심 토픽 정보 ===")
    print(topic_model.get_topic_info().head(10))
    
    # 모델 및 데이터 저장
    model_dir = os.path.join(PROCESSED_DIR, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    topic_model.save(os.path.join(model_dir, "bertopic_model"), serialization="safetensors", save_ctfidf=True)
    
    save_df_path = os.path.join(PROCESSED_DIR, 'documents_with_topics.pkl')
    df.to_pickle(save_df_path)
    print(f"\n✅ 완료! 토픽이 추가된 데이터가 {save_df_path} 에 저장됐어.")

if __name__ == "__main__":
    main()