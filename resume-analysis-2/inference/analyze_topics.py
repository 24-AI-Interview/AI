import pandas as pd
import os
from bertopic import BERTopic

# 1. 경로 설정
PROCESSED_DIR = 'data/processed/'
FILE_PATH = os.path.join(PROCESSED_DIR, 'documents_with_topics.pkl')
MODEL_PATH = os.path.join(PROCESSED_DIR, 'model', 'bertopic_model')

def main():
    if not os.path.exists(FILE_PATH):
        print("저장된 데이터가 없어!")
        return

    print("저장된 데이터 및 토픽 모델 불러오는 중...")
    df = pd.read_pickle(FILE_PATH)
    topic_model = BERTopic.load(MODEL_PATH)
    
    # 토픽 정보 가져오기 (Name 대신 Representation 사용!)
    topic_info = topic_model.get_topic_info()
    
    # -1번 토픽(노이즈) 제외
    df_valid = df[df['Topic'] != -1]
    
    print("\n📊 [토픽별 전문가 평가 분석 결과]")
    
    # 토픽별로 전체 개수와 그룹(High, Mid, Low)별 개수 계산
    topic_stats = df_valid.groupby(['Topic', 'Group']).size().unstack(fill_value=0)
    topic_stats['Total'] = topic_stats.sum(axis=1)
    
    # High, Low 비율 계산
    for group in ['High', 'Low']:
        if group in topic_stats.columns:
            topic_stats[f'{group}_Ratio(%)'] = (topic_stats[group] / topic_stats['Total'] * 100).round(1)
        else:
            topic_stats[f'{group}_Ratio(%)'] = 0.0

    # 🌟 Name 대신 Representation(진짜 단어 리스트)을 가져와서 병합
    topic_summary = topic_stats.merge(topic_info[['Topic', 'Representation']], on='Topic', how='left')
    
    # 의미 있는 분석을 위해 문서 수가 최소 30개 이상인 토픽만 필터링
    topic_summary = topic_summary[topic_summary['Total'] >= 30]

    # 1. 고득점 비율이 가장 높은 '필승 토픽' Top 5
    winning_topics = topic_summary.sort_values(by='High_Ratio(%)', ascending=False).head(5)
    print("\n🌟 [Winning Topics] 고득점(High) 비율이 높은 역량 Top 5")
    for idx, row in winning_topics.iterrows():
        # 리스트 형태의 단어들을 보기 좋게 쉼표로 연결
        keywords = ", ".join(row['Representation'][:7]) # 상위 7개 단어만 추출
        print(f"[{row['Topic']}번 토픽] 고득점 비율: {row['High_Ratio(%)']}% (총 {row['Total']}개)")
        print(f"👉 핵심 키워드: {keywords}\n")

    # 2. 저득점 비율이 가장 높은 '위험 토픽' Top 5
    redflag_topics = topic_summary.sort_values(by='Low_Ratio(%)', ascending=False).head(5)
    print("\n🚨 [Red Flag Topics] 저득점(Low) 비율이 높은 역량 Top 5")
    for idx, row in redflag_topics.iterrows():
        keywords = ", ".join(row['Representation'][:7])
        print(f"[{row['Topic']}번 토픽] 저득점 비율: {row['Low_Ratio(%)']}% (총 {row['Total']}개)")
        print(f"👉 핵심 키워드: {keywords}\n")

if __name__ == "__main__":
    main()