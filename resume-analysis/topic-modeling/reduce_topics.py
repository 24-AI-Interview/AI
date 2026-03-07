import pandas as pd
from bertopic import BERTopic

# 데이터 로드
df = pd.read_csv("../data/processed_resume_data.csv")
docs = df["Answer"].astype(str).tolist()

# 기존 BERTopic 모델 로드
topic_model = BERTopic.load("../data/bertopic_model_final")

print("기존 topic 개수:", len(topic_model.get_topic_info()) - 1)

# topic 줄이기
topic_model_reduced = topic_model.reduce_topics(
    docs,
    nr_topics=30
)

topic_info_reduced = topic_model_reduced.get_topic_info()

print("\n==============================")
print("Reduced Topic 결과")
print("==============================")
print("토픽 개수:", len(topic_info_reduced) - 1)

# 저장
topic_info_reduced.to_csv("../data/topic_info_reduced.csv", index=False)

topic_model_reduced.save("../data/bertopic_model_reduced")

print("\nReduced BERTopic 모델 저장 완료")