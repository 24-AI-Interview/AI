import pandas as pd
from bertopic import BERTopic

# 모델 로드 (89 topic 모델)
model_path = "../data/bertopic_model_final"
topic_model = BERTopic.load(model_path)

# topic 정보
topic_info = topic_model.get_topic_info()

print("\n===== Topic Overview =====")
print(topic_info.head(20))

print("\n===== Topic Keywords =====")

for topic_id in topic_info.Topic:

    if topic_id == -1:
        continue

    words = topic_model.get_topic(topic_id)

    # 상위 키워드만 출력
    keywords = [w for w, _ in words[:8]]

    print(f"\nTopic {topic_id}")
    print("keywords:", keywords)