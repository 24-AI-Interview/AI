## 📊 학습 과정 및 하이퍼파라미터 튜닝 기록

이 프로젝트의 핵심인 **4단계 점진적 미세 조정(Progressive Fine-Tuning)**의 상세한 과정은 별도의 문서로 정리해 두었습니다. 

단순히 코드를 돌린 것이 아니라, 학습 과정에서 발생한 과적합(Overfitting)과 언더피팅(Underfitting) 문제를 어떻게 진단하고 해결했는지 구체적인 고민의 흔적을 담았습니다.

* **주요 내용:**
  * 단계별 하이퍼파라미터 (Epoch, Learning Rate, Weight Decay 등) 설정 값
  * 특정 레이어(Backbone, Mixed_7a, Mixed_6a 등)를 동결 및 해제한 이유
  * 데이터 증강(Augmentation) 전략 변경 사유
  * 각 단계별 Train/Val Accuracy 및 Loss 변화 추이

👉 **[상세 학습 과정 보러 가기 (Notion 링크)](https://www.notion.so/ai-tuning-2f8c620638bf80198498e4cfb66a6849)**