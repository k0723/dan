from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import hdbscan
import numpy as np
import re
import time
from collections import defaultdict
from keybert import KeyBERT

# 모델 초기화
model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model='all-MiniLM-L6-v2')

# 조사 제거용 리스트
POSTPOSITIONS = ["은", "는", "이", "가", "을", "를", "에", "에서", "에게", "로", "으로", "과", "와", "도", "만", "까지", "부터"]

# 조사 제거 함수
def clean_postpositions(word: str) -> str:
    return re.sub(f"({'|'.join(POSTPOSITIONS)})$", '', word)

# LLM 백업 키워드 함수
def fallback_keyword_with_llm(text: str) -> str:
    prompt = f"""
    문장에서 핵심 키워드 하나만 뽑아줘. 불필요한 조사는 제거하고, 단어 하나로만 반환해:
    문장: "{text}"
    키워드:
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ LLM 실패:", e)
        return ""

# 키워드 추출 함수

def extract_keywords(texts: list[str], timeout: float = 3.0) -> list[str]:
    raw_keywords = []

    for text in texts:
        try:
            print("🔍 입력 문장:", text)
            start_time = time.time()

            keywords = kw_model.extract_keywords(text, top_n=1)
            print("🧠 KeyBERT 결과:", keywords)
            if not keywords or not keywords[0]:
                raise ValueError("kw_model 실패")

            raw_keyword = keywords[0][0]
            cleaned = clean_postpositions(raw_keyword)
            print("🧼 조사 제거 후:", cleaned)

            if len(cleaned) < 2 or not cleaned.isalnum():
                raise ValueError("키워드 너무 짧거나 유효하지 않음")

            if time.time() - start_time > timeout:
                raise TimeoutError("키워드 추출 시간 초과")

            raw_keywords.append(cleaned)

        except Exception as e:
            print("⚠️ 예외 발생:", e)
            keyword = fallback_keyword_with_llm(text)
            cleaned = clean_postpositions(keyword)
            print("🤖 LLM 대체 키워드:", cleaned)
            if cleaned:
                raw_keywords.append(cleaned)

    # ✅ 조사 제거 후 중복 키워드 제거
    unique_keywords = []
    seen = set()
    for kw in raw_keywords:
        base = clean_postpositions(kw)
        if base and base not in seen:
            seen.add(base)
            unique_keywords.append(base)

    return unique_keywords


# 전체 분석 함수 (클러스터링 + 키워드)
def analyze_texts(texts: list[str]) -> dict:
    # 1. 임베딩
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = normalize(embeddings)

    # 2. HDBSCAN 군집화
    best_score = -1
    best_labels = None
    best_size = None
    for size in range(2, min(10, len(texts))):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=size)
        labels = clusterer.fit_predict(embeddings)

        if len(set(labels)) <= 1 or all(l == -1 for l in labels):
            continue

        try:
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_labels = labels
                best_size = size
        except:
            continue

    # 3. 군집 실패 시 fallback
    if best_labels is None:
        return {
            "clusters": {
                f"Text {i}": {
                    "summary": text[:100] + "...",
                    "cluster": -1
                } for i, text in enumerate(texts)
            },
            "keywords": extract_keywords(texts)
        }

    # 4. 군집 결과 구성
    clusters = defaultdict(list)
    valid_texts = []
    for text, label in zip(texts, best_labels):
        clusters[label].append(text)
        if label != -1:
            valid_texts.append(text)

    cluster_info = {}
    for label, cluster_texts in clusters.items():
        summary = cluster_texts[0][:100] + "..."
        cluster_info[f"Cluster {label}"] = {
            "count": len(cluster_texts),
            "summary": summary,
            "min_cluster_size": best_size
        }

    # 5. 키워드 추출 (클러스터링 된 텍스트만)
    keywords = extract_keywords(valid_texts)

    return {
        "clusters": cluster_info,
        "keywords": keywords
    }
