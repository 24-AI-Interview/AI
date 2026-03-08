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

    df = pd.read_pickle(FILE_PATH)
    topic_model = BERTopic.load(MODEL_PATH)
    topic_info = topic_model.get_topic_info()
    
    # -1번 토픽(노이즈) 제외
    df_valid = df[df['Topic'] != -1]
    
    print("\n📊 [점수 그룹별 가장 많이 등장한 토픽 Top 5]")
    
    # High, Mid, Low 그룹별로 분석
    groups = ['High', 'Mid', 'Low']
    
    for group in groups:
        group_df = df_valid[df_valid['Group'] == group]
        if group_df.empty:
            continue
            
        total_in_group = len(group_df)
        print(f"\n========================================")
        print(f"🎯 [{group} 그룹] 총 {total_in_group}개 답변의 주요 토픽")
        print(f"========================================")
        
        # 해당 그룹 내에서 가장 많이 나온 토픽 Top 5 추출
        top_topics = group_df['Topic'].value_counts().head(5)
        
        for topic, count in top_topics.items():
            # 그룹 내 차지하는 비중(%)
            ratio = round(count / total_in_group * 100, 1)
            
            # 토픽 키워드 가져오기
            representation = topic_info[topic_info['Topic'] == topic]['Representation'].values[0]
            keywords = ", ".join(representation[:7])
            
            print(f"[{topic}번 토픽] 비중: {ratio}% (해당 그룹 내 {count}개)")
            print(f"👉 핵심 키워드: {keywords}\n")

if __name__ == "__main__":
    main()