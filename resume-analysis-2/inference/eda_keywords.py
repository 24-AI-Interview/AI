import pandas as pd
import os
from collections import Counter
from kiwipiepy import Kiwi

# 1. 자바가 필요 없는 Kiwi 형태소 분석기 초기화
kiwi = Kiwi()
PROCESSED_DIR = 'data/processed/'

def extract_context_keywords(text_list):
    all_phrases = []
    
    for text in text_list:
        if not isinstance(text, str):
            continue
            
        # 노이즈(글자수 등) 사전 제거
        text = text.replace('글자수', '').replace('Byte', '')
        
        tokens = kiwi.tokenize(text)
        words = []
        
        for t in tokens:
            # NNG(일반명사), NNP(고유명사), VV(동사), VA(형용사), XR(어근) 추출
            if t.tag in ['NNG', 'NNP', 'VV', 'VA', 'XR'] and len(t.form) > 1:
                # 동사와 형용사는 '다'를 붙여 기본형 느낌으로 복원
                if t.tag in ['VV', 'VA']:
                    words.append(t.form + "다")
                else:
                    words.append(t.form)
        
        # Bi-gram (두 단어씩 이어 붙여서 맥락 만들기)
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            all_phrases.append(phrase)
            
    return all_phrases

def main():
    print("🥝 Kiwi 형태소 분석기를 활용한 명사 추출 및 EDA 시작...\n")
    
    for group in ['high', 'low']:
        file_path = os.path.join(PROCESSED_DIR, f'{group}_quality.csv') 
        if not os.path.exists(file_path):
            continue
            
        df = pd.read_csv(file_path)
        
        answers = df['Answer'].dropna().tolist()
        reviews = df['Expert_Review'].dropna().tolist()
        
        print(f"[{group.upper()} 그룹 분석 중...]")
        # 명사 추출
        answer_nouns = extract_context_keywords(answers)
        review_nouns = extract_context_keywords(reviews)
        
        # 빈도수 상위 20개 추출 (조금 더 늘려봤어!)
        top_answer_keywords = Counter(answer_nouns).most_common(20)
        top_review_keywords = Counter(review_nouns).most_common(20)
        
        print(f"✅ 지원자 답변 빈도 Top 20:")
        print([word for word, count in top_answer_keywords])
        print(f"✅ 전문가 리뷰 빈도 Top 20:")
        print([word for word, count in top_review_keywords])
        print("-" * 50)

if __name__ == "__main__":
    main()