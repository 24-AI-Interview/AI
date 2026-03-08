import pandas as pd
import os
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

PROCESSED_DIR = 'data/processed/'
FILE_PATH = os.path.join(PROCESSED_DIR, 'documents_with_topics.pkl')
MODEL_PATH = os.path.join(PROCESSED_DIR, 'model', 'bertopic_model')

def custom_tokenizer(text):
    """정규식 없이 무조건 띄어쓰기 기준으로만 단어를 자르는 함수"""
    return text.split()

def main():
    if not os.path.exists(FILE_PATH):
        print("데이터 파일이 없어!")
        return

    print("저장된 데이터 및 토픽 모델 불러오는 중...")
    df = pd.read_pickle(FILE_PATH)
    topic_model = BERTopic.load(MODEL_PATH)

    # 1. 한국어가 절대 증발하지 않도록 띄어쓰기 기준 Vectorizer 강제 적용
    vectorizer_model = CountVectorizer(tokenizer=custom_tokenizer, lowercase=False)

    print("🚀 잃어버린 한글 키워드 재추출 중 (10초 컷!)...")
    # 2. 클러스터링 결과는 유지한 채, 명사/동사만 남겨둔 Tokenized_Doc으로 이름표만 업데이트!
    docs = df['Tokenized_Doc'].tolist()
    topic_model.update_topics(docs, vectorizer_model=vectorizer_model)
    
    # 3. 모델 다시 저장
    topic_model.save(MODEL_PATH, serialization="safetensors", save_ctfidf=True)

    print("\n✅ 키워드 복구 완료! 이제 analyze_topics.py 를 다시 실행해봐!")

if __name__ == "__main__":
    main()