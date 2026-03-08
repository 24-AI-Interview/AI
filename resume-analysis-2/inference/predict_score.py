import pandas as pd
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from kiwipiepy import Kiwi

# 1. 경로 설정
PROCESSED_DIR = 'data/processed/'
DF_PATH = os.path.join(PROCESSED_DIR, 'documents_with_topics.pkl')
MODEL_PATH = os.path.join(PROCESSED_DIR, 'model', 'bertopic_model')

# 2. 형태소 분석기 초기화
kiwi = Kiwi()

def get_keywords(text):
    """학습할 때와 완벽히 동일한 기준으로 명사/동사/형용사만 추출"""
    if not isinstance(text, str) or not text.strip(): return ""
    tokens = kiwi.tokenize(text)
    result = [t.form + ('다' if t.tag in ['VV', 'VA'] else '') 
              for t in tokens if t.tag in ['NNG', 'NNP', 'VV', 'VA'] and len(t.form) > 1]
    return " ".join(result)

def main():
    print("🧠 AI 평가 모델 및 SBERT 임베딩 로딩 중...")
    
    # 저장된 통계 데이터 및 토픽 모델 로드
    df = pd.read_pickle(DF_PATH)
    topic_model = BERTopic.load(MODEL_PATH)
    
    # 임베딩 모델 로드 (학습 때 썼던 것과 동일한 모델)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', device=device)

    # 3. 토픽별 통계(Scoring Table) 사전 계산
    df_valid = df[df['Topic'] != -1]
    topic_stats = df_valid.groupby(['Topic', 'Group']).size().unstack(fill_value=0)
    topic_stats['Total'] = topic_stats.sum(axis=1)
    
    # ---------------------------------------------------------
    # 📝 [테스트] 사용자가 입력한 가상의 새로운 자소서
    # ---------------------------------------------------------
    test_resume = """
    저는 대학 시절 데이터 분석 동아리에서 팀장으로 활동하며, 
    팀원들 간의 의견 충돌이 발생했을 때 객관적인 데이터를 바탕으로 
    회의를 주도하여 갈등을 해결하고 프로젝트를 성공적으로 마친 경험이 있습니다.
    """
    
    print("\n" + "="*50)
    print("📝 [사용자 입력 자소서]")
    print(test_resume.strip())
    print("="*50)
    
    # 4. 예측 파이프라인
    # 4-1. 텍스트 토큰화 (Kiwi)
    tokenized_text = get_keywords(test_resume)
    
    # 4-2. 원본 문맥 임베딩 (SBERT)
    embedding = embed_model.encode([test_resume]) 
    
    # 4-3. 어떤 토픽인지 예측 (Transform)
    topics, probs = topic_model.transform([tokenized_text], embedding)
    pred_topic = topics[0]
    
    print(f"\n🔍 [AI 스코어링 분석 결과]")
    if pred_topic == -1:
        print("⚠️ 이 자소서는 기존 데이터의 어떤 패턴에도 속하지 않는 독특한(또는 내용이 부족한) 글입니다.")
        return
        
    # 5. 스코어링 계산
    topic_info = topic_model.get_topic_info()
    representation = topic_info[topic_info['Topic'] == pred_topic]['Representation'].values[0]
    keywords = ", ".join(representation[:7])
    
    stats = topic_stats.loc[pred_topic]
    total = stats['Total']
    
    high_pct = round((stats.get('High', 0) / total) * 100, 1)
    mid_pct = round((stats.get('Mid', 0) / total) * 100, 1)
    low_pct = round((stats.get('Low', 0) / total) * 100, 1)
    
    # 점수 산출 (High 100점, Mid 50점, Low 0점 기준 가중 평균)
    expected_score = round((high_pct * 1.0) + (mid_pct * 0.5) + (low_pct * 0.0), 1)
    
    print(f"✅ 매핑된 핵심 역량 (토픽 {pred_topic}): [{keywords}]")
    print(f"📊 이 역량을 어필한 과거 지원자들의 실제 평가 분포:")
    print(f"   - 🟢 우수 (High): {high_pct}%")
    print(f"   - 🟡 보통 (Mid) : {mid_pct}%")
    print(f"   - 🔴 미흡 (Low) : {low_pct}%")
    print(f"\n🏆 최종 예상 경쟁력 점수: {expected_score} / 100점")

if __name__ == "__main__":
    main()