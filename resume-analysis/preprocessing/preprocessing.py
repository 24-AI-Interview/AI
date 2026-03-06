import pandas as pd

# 경로
input_path = "../data/jobkorea_contents_final_with_review.csv"
output_path = "../data/processed_resume_data.csv"

# 1. 데이터 로드
df = pd.read_csv(input_path)

print("원본 데이터 개수:", len(df))

# 2. 결측 제거
df = df.dropna(subset=["Answer", "Question"])

# 3. 줄바꿈 / 공백 정리
df["Answer"] = df["Answer"].str.replace("\n", " ", regex=False).str.strip()
df["Question"] = df["Question"].str.replace("\n", " ", regex=False).str.strip()

# 4. 너무 짧은 답변 제거
df = df[df["Answer"].str.len() > 30]

# 5. 모델 입력 텍스트 생성
df["text"] = df["Question"] + " " + df["Answer"]

print("전처리 후 데이터 개수:", len(df))

# 6. 필요한 컬럼만 유지
df_processed = df[[
    "ID",
    "Company",
    "JobRole",
    "Question",
    "Answer",
    "text"
]]

# 7. 저장
df_processed.to_csv(output_path, index=False)

print("저장 완료:", output_path)