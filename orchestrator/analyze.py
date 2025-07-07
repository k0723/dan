from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import hdbscan
import numpy as np
import re
import time
from collections import defaultdict
from keybert import KeyBERT
from agents.agent_launcher import launch_agents

# 모델 초기화
model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model='all-MiniLM-L6-v2')
SIM_THRESHOLD = 0.7
# 조사 제거용 리스트
POSTPOSITIONS = ["은", "는", "이", "가", "을", "를", "에", "에서", "에게", "로", "으로", "과", "와", "도", "만", "까지", "부터", "의"]

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

def extract_keywords(texts: list[str]) -> list[str]:
    """
    각 문장에서 KeyBERT로 상위 1개 키워드를 추출하고,
    조사 제거 후 반환합니다.
    """
    keywords = []
    for text in texts:
        try:
            # KeyBERT로 top_n=1 키워드 뽑기
            result = kw_model.extract_keywords(text, top_n=1)
            if not result:
                raise ValueError("키워드 없음")
            raw_kw = result[0][0]
            # 조사 제거
            cleaned = clean_postpositions(raw_kw)
            keywords.append(cleaned)
        except Exception as e:
            # KeyBERT 실패 시 LLM 백업 호출
            fb = fallback_keyword_with_llm(text)
            cleaned_fb = clean_postpositions(fb)
            if cleaned_fb:
                keywords.append(cleaned_fb)
            else:
                # LLM도 실패하면 빈 문자열 또는 기본값
                keywords.append("")
    return keywords

# 전체 분석 함수 (클러스터링 + 키워드)
def analyze_texts(texts: list[str]) -> dict:
    # 1) 문장별 키워드 추출
    raw_keywords = extract_keywords(texts)
    # 순서 유지 중복 제거
    unique_keywords = list(dict.fromkeys(raw_keywords))
    if not unique_keywords:
        return {"clusters": {}, "keywords": []}

    # 2) 키워드 임베딩
    kw_embeddings = model.encode(unique_keywords, show_progress_bar=False)
    kw_embeddings = normalize(kw_embeddings)

    # 3) HDBSCAN으로 키워드 클러스터링
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    kw_labels = clusterer.fit_predict(kw_embeddings)

    # 4) 클러스터 정보 구성
    clusters = defaultdict(list)
    for kw, label in zip(unique_keywords, kw_labels):
        clusters[label].append(kw)

    cluster_info = {}
    for label, kws in clusters.items():
        if label == -1:
            # 노이즈 클러스터는 건너뛰거나 따로 처리
            continue
        # 대표 키워드: 해당 클러스터 내에서 가장 빈번한(여기선 첫 번째) 키워드
        rep = kws[0]
        cluster_info[f"Cluster {label}"] = {
            "keywords": kws,
            "representative": rep,
            "count": len(kws)
        }
    launch_agents(unique_keywords)
    # 5) 결과 반환
    return {
        "clusters": cluster_info,
        "keywords": unique_keywords
    }


def group_by_similarity(texts: list[str], threshold: float = SIM_THRESHOLD) -> dict:
    embeddings = model.encode(texts, convert_to_tensor=True)
    visited = set()
    clusters = defaultdict(list)
    cluster_id = 0

    for i in range(len(texts)):
        if i in visited:
            continue
        clusters[cluster_id].append(texts[i])
        visited.add(i)

        for j in range(i + 1, len(texts)):
            if j in visited:
                continue
            sim = util.cos_sim(embeddings[i], embeddings[j]).item()
            if sim >= threshold:
                clusters[cluster_id].append(texts[j])
                visited.add(j)

        cluster_id += 1

    return dict(clusters)