# preprocessing/prepare_documents.py
import pandas as pd
import os
import re

RAW_DATA_PATH = 'data/raw/jobkorea_contents_final_with_review.csv'
PROCESSED_DIR = 'data/processed/'

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 꼬리표 및 노이즈 정제
    text = re.sub(r'(아쉬운점|좋은점)\s*\d+', '', text)
    text = re.sub(r'글자수\s*[\d,]+자', '', text)
    text = re.sub(r'[\d,]+Byte', '', text)
    
    return text.strip()

def main():
    print("데이터 로딩 및 정제 시작...")
    df = pd.read_csv(RAW_DATA_PATH)

    # 전문가 평점이 있는 데이터만 필터링
    df = df[df['Expert_Rating'].notna()].copy()
    df['Expert_Rating'] = pd.to_numeric(df['Expert_Rating'], errors='coerce')
    df = df[df['Expert_Rating'].notna()].copy()

    # 점수 그룹화 (1~2: Low, 3: Mid, 4~5: High)
    def categorize_rating(rating):
        if rating <= 2: return 'Low'
        elif rating == 3: return 'Mid'
        elif rating >= 4: return 'High'
        else: return 'Unknown'
    
    df['Group'] = df['Expert_Rating'].apply(categorize_rating)

    # 문항(Question)과 답변(Answer) 결합하여 하나의 Document 생성
    df['Question'] = df['Question'].fillna('')
    df['Answer'] = df['Answer'].fillna('')
    df['Document'] = df['Question'] + " " + df['Answer']

    # 텍스트 노이즈 정제
    df['Document'] = df['Document'].apply(clean_text)
    df['Expert_Review'] = df['Expert_Review'].apply(clean_text)

    # 필요한 컬럼만 추출
    final_df = df[['Group', 'Expert_Rating', 'Document', 'Expert_Review']]
    
    # 저장
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    save_path = os.path.join(PROCESSED_DIR, 'documents_prepared.csv')
    final_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✅ 정제 완료. 저장 경로: {save_path} (총 {len(final_df)}건)")

if __name__ == "__main__":
    main()