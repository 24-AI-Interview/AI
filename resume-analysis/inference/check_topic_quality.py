import pandas as pd
from bertopic import BERTopic

# ==============================
# 모델 로드
# ==============================
model_path = "../data/bertopic_model_reduced"
topic_model = BERTopic.load(model_path)

# ==============================
# Topic 정보 출력
# ==============================
topic_info = topic_model.get_topic_info()

print("\n==============================")
print("Topic Overview")
print("==============================")
print(topic_info.head(20))


# ==============================
# Topic keywords 확인
# ==============================
print("\n==============================")
print("Topic Keywords")
print("==============================")

for topic_id in topic_info.Topic[:10]:   # 상위 10개 topic
    if topic_id == -1:
        continue

    print(f"\nTopic {topic_id}")
    print(topic_model.get_topic(topic_id))


# ==============================
# 실제 문서 샘플 확인
# ==============================
print("\n==============================")
print("Representative Documents")
print("==============================")

rep_docs = topic_model.get_representative_docs()

for topic_id in list(rep_docs.keys())[:10]:
    print(f"\nTopic {topic_id}")

    docs = rep_docs[topic_id]

    for d in docs[:2]:   # topic당 2개만 출력
        print("-" * 40)
        print(d[:300])