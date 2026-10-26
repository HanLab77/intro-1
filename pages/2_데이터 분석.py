import base64
from collections import Counter
import html
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.title("📊 데이터 분석 교과서 실험실")
st.caption("고등학교 데이터 분석 교과서 목차를 바탕으로 한 동적 탐구 웹페이지")

st.markdown(
    """
    <style>
    .main { background: linear-gradient(180deg, #f5f7f1 0%, #eef5f3 100%); }
    .section-header {
        background: #1d4d3b;
        color: white;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .subbox {
        border-left: 4px solid #8db36b;
        background: rgba(141,179,107,0.08);
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.2rem 0 0.9rem 0;
    }
    .note {
        background: #edf4ff;
        border: 1px solid #d5e4ff;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        color: #224a72;
    }
    .kicker {
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #4a6b3a;
        font-weight: 700;
    }
    /* 상단 메뉴(단계 탭)만 꾸민다. 단계 안쪽의 작은 탭은 기본 모양 그대로 둔다. */
    .st-key-step_menu [role="tablist"]:not([data-testid="stTabPanel"] [role="tablist"]) {
        gap: 0.5rem;
        flex-wrap: wrap;
        height: auto;
        padding: 0.6rem;
        background: #f4f7f2;
        border: 1px solid #dce6d8;
        border-radius: 16px;
        margin-bottom: 0.6rem;
    }
    .st-key-step_menu [role="tablist"]:not([data-testid="stTabPanel"] [role="tablist"]) > [role="tab"] {
        height: auto;
        padding: 0.6rem 1.1rem;
        border-radius: 999px;
        background: #ffffff;
        border: 1.5px solid #c9dcc6;
        color: #1d4d3b;
        transition: all 0.15s ease;
    }
    .st-key-step_menu [role="tablist"]:not([data-testid="stTabPanel"] [role="tablist"]) > [role="tab"] p { font-size: 1.12rem; font-weight: 700; margin: 0; color: inherit; }
    .st-key-step_menu [role="tablist"]:not([data-testid="stTabPanel"] [role="tablist"]) > [role="tab"][data-hovered] { background: #e6f0e3; transform: translateY(-1px); }
    .st-key-step_menu [role="tablist"]:not([data-testid="stTabPanel"] [role="tablist"]) > [role="tab"][data-selected] {
        background: #1d4d3b;
        border-color: #1d4d3b;
        color: #ffffff;
        box-shadow: 0 3px 10px rgba(29, 77, 59, 0.3);
    }
    /* 5번째부터는 데이터 분석 메뉴: 주황 계열로 단계 탭과 구분한다. */
    .st-key-step_menu [role="tablist"]:not([data-testid="stTabPanel"] [role="tablist"]) > [role="tab"]:nth-of-type(n+5) { background: #fff8ef; border-color: #f0d3ad; color: #8a4b1c; }
    .st-key-step_menu [role="tablist"]:not([data-testid="stTabPanel"] [role="tablist"]) > [role="tab"]:nth-of-type(n+5)[data-hovered] { background: #fdebd5; }
    .st-key-step_menu [role="tablist"]:not([data-testid="stTabPanel"] [role="tablist"]) > [role="tab"]:nth-of-type(n+5)[data-selected] {
        background: #b66b3d;
        border-color: #b66b3d;
        color: #ffffff;
        box-shadow: 0 3px 10px rgba(182, 107, 61, 0.3);
    }
    .st-key-step_menu [role="tablist"]:not([data-testid="stTabPanel"] [role="tablist"]) .react-aria-SelectionIndicator { display: none; }
    .st-key-site_finder [data-testid="stExpander"] > details {
        border: 2px solid #1d4d3b;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(29, 77, 59, 0.15);
    }
    .st-key-site_finder [data-testid="stExpander"] > details > summary {
        background: linear-gradient(90deg, #e3f0e7 0%, #f3f8ee 100%);
        padding: 0.9rem 1.1rem;
        border-radius: 10px;
    }
    .st-key-site_finder [data-testid="stExpander"] > details > summary:hover { background: #d4e8da; }
    .st-key-site_finder [data-testid="stExpander"] > details > summary p {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1d4d3b;
    }
    /* 💻 코드 실습: 코드 셀은 고정폭 글꼴, 파일 패널은 연한 배경 */
    [class*="st-key-nb_cell_"] textarea { font-family: 'DM Mono', Consolas, monospace; font-size: 0.92rem; background: #f8faf7; }
    .st-key-nb_file_panel { background: #f4f7f2; }
    [class*="st-key-nb_out_"] { border-top: 1px dashed #d5dfd2; padding-top: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

TEXTBOOK_CHAPTERS = {
    "데이터 준비와 분석": [
        "01. 데이터 수집: 문제 정의와 데이터 수집, 수집한 데이터의 특성",
        "02. 데이터 전처리와 시각화: 데이터 전처리, 데이터 시각화",
        "03. 데이터 분석 방법: 탐색적 데이터 분석",
    ],
    "데이터 모델링과 평가": [
        "01. 데이터 모델의 개념: 데이터 모델의 이해, 데이터 분석",
        "02. 회귀 분석: 회귀 분석의 이해, 통계적 회귀 분석, 기계학습 회귀 분석, 회귀 분석 성능 평가",
        "03. 군집 분석: 군집 분석의 이해, k-평균 군집 분석, k-평균 군집 분석 성능 평가",
        "04. 연관 분석: 연관 분석의 이해, 연관 분석 과정",
    ],
}

with st.sidebar:
    st.markdown("## 학습 목차")
    selected_chapter = st.radio(
        "단원 선택",
        ["데이터 준비와 분석", "데이터 모델링과 평가"],
        horizontal=False,
    )
    st.markdown("### 세부 항목")
    for item in TEXTBOOK_CHAPTERS[selected_chapter]:
        st.markdown(f"- {item}")
    st.divider()
    st.markdown("### 빠른 실험")
    st.button("데이터 새로고침", width="stretch")


@st.cache_data
def load_gdp_data():
    df = pd.read_csv("data/gdp_data.csv")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"Country Name": "국가", "Country Code": "국가코드", "Indicator Name": "지표명"})
    return df


@st.cache_data
def make_gdp_sample():
    df = load_gdp_data()
    year_cols = [c for c in df.columns if c.isdigit() and int(c) >= 2000]
    long_df = df.melt(id_vars=["국가", "국가코드"], value_vars=year_cols, var_name="연도", value_name="GDP(US$)")
    long_df["연도"] = long_df["연도"].astype(int)
    return long_df


@st.cache_data
def make_student_sample():
    rng = np.random.default_rng(42)
    n = 60
    study = rng.normal(3, 1.2, n).clip(0.2).round(1)
    sleep = rng.normal(7, 1, n).round(1)
    phone = rng.normal(4, 1.5, n).clip(0.5).round(1)
    score = (50 + study * 8 - phone * 2 + (sleep - 7) * 2 + rng.normal(0, 6, n)).clip(0, 100).round()
    df = pd.DataFrame({
        "학생ID": [f"S{i + 1:03d}" for i in range(n)],
        "학년": rng.choice(["1학년", "2학년", "3학년"], n),
        "성별": rng.choice(["남", "여"], n),
        "공부시간": study,
        "수면시간": sleep,
        "스마트폰사용": phone,
        "시험점수": score,
    })
    # 전처리 실습을 위해 결측값, 이상치, 중복 행을 일부러 넣어 둔다.
    df.loc[[3, 17, 29], "수면시간"] = np.nan
    df.loc[[8, 41], "시험점수"] = np.nan
    df.loc[22, "학년"] = np.nan
    df.loc[50, "스마트폰사용"] = 19.0
    return pd.concat([df, df.iloc[[5, 12]]], ignore_index=True)


SAMPLE_DATASETS = {
    "학생 생활 설문 (결측값·이상치 포함)": make_student_sample,
    "국가별 GDP 2000~2022 (World Bank)": make_gdp_sample,
}


def read_uploaded_csv(uploaded_file):
    data = uploaded_file.getvalue()
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("UTF-8 또는 CP949로 저장된 CSV 파일을 사용해 주세요.")


def is_numeric(series):
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def split_columns(df):
    num_cols = [c for c in df.columns if is_numeric(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols


def column_profile(df):
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "열 이름": col,
            "자료형": str(s.dtype),
            "종류": "수치형" if is_numeric(s) else "범주형",
            "결측값": int(s.isna().sum()),
            "고유값 수": int(s.nunique()),
            "예시 값": str(s.dropna().iloc[0]) if s.notna().any() else "",
        })
    return pd.DataFrame(rows)


def fld(col):
    # Altair는 필드 이름의 . [ ] 를 경로로 해석하므로 이스케이프한다.
    return str(col).replace(".", "\\.").replace("[", "\\[").replace("]", "\\]")


def compute_regression(df):
    x = df["학습시간"]
    y = df["시험점수"]
    x_mean = x.mean()
    y_mean = y.mean()
    slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
    intercept = y_mean - slope * x_mean
    return slope, intercept


def kmeans(points, k, max_iter=10):
    centers = [points[i] for i in range(k)]
    for _ in range(max_iter):
        groups = [[] for _ in range(k)]
        for idx, point in enumerate(points):
            dists = [((point[0] - c[0]) ** 2 + (point[1] - c[1]) ** 2) ** 0.5 for c in centers]
            cluster_id = min(range(len(dists)), key=dists.__getitem__)
            groups[cluster_id].append(idx)

        new_centers = []
        for group in groups:
            if not group:
                new_centers.append(centers[len(new_centers)])
                continue
            xs = [points[i][0] for i in group]
            ys = [points[i][1] for i in group]
            new_centers.append((sum(xs) / len(xs), sum(ys) / len(ys)))
        if new_centers == centers:
            break
        centers = new_centers

    labels = []
    for point in points:
        dists = [((point[0] - c[0]) ** 2 + (point[1] - c[1]) ** 2) ** 0.5 for c in centers]
        labels.append(min(range(len(dists)), key=dists.__getitem__))
    return labels


# 데이터 수집하기(4-1차시) 자료에 소개된 공개 데이터 제공 사이트
PUBLIC_DATA_SITES = [
    {"사이트": "공공데이터포털", "분류": "공공·국가 데이터", "아이콘": "🏛️", "주소": "https://www.data.go.kr",
     "검색": "https://www.data.go.kr/tcs/dss/selectDataSetList.do?keyword={q}",
     "설명": "정부와 공공기관이 생성·취득한 데이터를 주제별로 한곳에서 제공하는 통합 창구",
     "예시": "소상공인시장진흥공단 상가정보, 주차장·지하철 정보", "제공형태": "파일, 오픈 API",
     "키워드": "공공 정부 상가 교통 지하철 주차 아파트 관광 행정 API"},
    {"사이트": "국가통계포털 KOSIS", "분류": "공공·국가 데이터", "아이콘": "📈", "주소": "https://kosis.kr",
     "검색": "https://kosis.kr/search/search.do?query={q}",
     "설명": "국내·국제·북한의 주요 통계를 한곳에 모아 원하는 통계를 쉽게 찾도록 제공",
     "예시": "주민등록 인구수, 출생아 수, 고용률, 과실 재배 면적", "제공형태": "통계표 다운로드, 오픈 API",
     "키워드": "통계 인구 출생 고용 임금 수면 농림 과실 사과 재배"},
    {"사이트": "KOSIS 100대 지표", "분류": "공공·국가 데이터", "아이콘": "💯", "주소": "https://kosis.kr/visual/nsportalStats/main.do",
     "검색": None,
     "설명": "여러 분야의 핵심 정보를 쉽게 파악하도록 통계청이 고른 100대 지표",
     "예시": "인구, 경제성장률, 물가 등 핵심 지표", "제공형태": "시각화 지표",
     "키워드": "지표 통계 인구 경제 물가 핵심"},
    {"사이트": "마이크로데이터 통합서비스 MDIS", "분류": "공공·국가 데이터", "아이콘": "🔬", "주소": "https://mdis.kostat.go.kr",
     "검색": None,
     "설명": "통계조사 원자료에서 개인정보·오류를 정리한 개별 단위(개인·가구) 데이터 제공 (회원가입 필요)",
     "예시": "청소년도박실태조사, 청소년건강패널조사, 가계금융복지조사", "제공형태": "CSV·텍스트 다운로드",
     "키워드": "마이크로데이터 설문 조사 청소년 가구 소득 건강 원자료"},
    {"사이트": "원윈도우", "분류": "공공·국가 데이터", "아이콘": "🪟", "주소": "https://www.data1window.kr",
     "검색": None,
     "설명": "공공·민간 데이터를 한 번에 검색하고 여러 빅데이터 플랫폼으로 연결해 주는 국가데이터통합플랫폼",
     "예시": "학교등급별 급식 잔반량 데이터", "제공형태": "통합 검색 → 각 플랫폼 연결",
     "키워드": "통합 검색 급식 잔반 학교 빅데이터 민간"},
    {"사이트": "통계지리정보서비스 SGIS", "분류": "지역·지도", "아이콘": "🗺️", "주소": "https://sgis.kostat.go.kr",
     "검색": None,
     "설명": "통계 정보와 지리 정보를 결합해 지도로 볼 수 있는 데이터 제공",
     "예시": "지역별 인구와 주택 통계, 업종 통계지도", "제공형태": "통계지도, 파일, API",
     "키워드": "지도 지역 인구 주택 상권 업종 동네"},
    {"사이트": "기상자료개방포털", "분류": "날씨·환경·농업", "아이콘": "🌦️", "주소": "https://data.kma.go.kr",
     "검색": None,
     "설명": "기상청이 모든 날씨 데이터를 한곳에서 찾을 수 있도록 제공 ([데이터]-[기상관측]-[지상])",
     "예시": "기온, 강수량, 풍속, 지면온도", "제공형태": "CSV·Excel, 대용량 API",
     "키워드": "날씨 기온 강수량 풍속 기후 폭우 폭염 환경"},
    {"사이트": "과수생육·품질관리시스템", "분류": "날씨·환경·농업", "아이콘": "🍎", "주소": "https://fruit.nihhs.go.kr",
     "검색": None,
     "설명": "농촌진흥청 국립원예특작과학원의 과수 생육·품질·기상·재배지 변동 정보",
     "예시": "사과 과중·당도·경도, 재배 면적 지도", "제공형태": "표, 그래프, 지도",
     "키워드": "사과 과수 농업 품질 당도 재배 기후변화"},
    {"사이트": "농식품 빅데이터 거래소 KADX", "분류": "날씨·환경·농업", "아이콘": "🥗", "주소": "https://kadx.co.kr",
     "검색": None,
     "설명": "농식품 분야 데이터를 사고팔거나 무료로 내려받을 수 있는 데이터 거래소",
     "예시": "학교등급별 급식 잔반량 데이터(무료)", "제공형태": "파일 데이터",
     "키워드": "농식품 급식 잔반 음식 소비 트렌드"},
    {"사이트": "문화셈터 (문화체육관광 통계포털)", "분류": "생활·건강·문화", "아이콘": "🏃", "주소": "https://stat.mcst.go.kr",
     "검색": None,
     "설명": "문화·예술·관광·체육 분야 통계 ([통계 자료실]-[체육]-[국민체력측정통계])",
     "예시": "국민체력측정통계(신체질량지수 BMI), 국민여행조사", "제공형태": "통계표, 차트, 다운로드",
     "키워드": "체육 체력 비만 BMI 건강 관광 문화 예술"},
    {"사이트": "캐글 Kaggle", "분류": "경진대회·해외", "아이콘": "🏆", "주소": "https://www.kaggle.com",
     "검색": "https://www.kaggle.com/datasets?search={q}",
     "설명": "전 세계 데이터 과학자들이 데이터셋과 코드를 공유하는 머신러닝 경진대회 플랫폼",
     "예시": "FIFA 선수 데이터, 타이타닉 데이터", "제공형태": "CSV 등 파일, 노트북",
     "키워드": "머신러닝 인공지능 경진대회 해외 스포츠 축구 영어"},
    {"사이트": "데이콘 DACON", "분류": "경진대회·해외", "아이콘": "🥇", "주소": "https://dacon.io",
     "검색": None,
     "설명": "국내 AI·데이터 분석 경진대회(해커톤) 플랫폼",
     "예시": "버스 운행 시간 예측", "제공형태": "대회용 데이터 파일",
     "키워드": "경진대회 해커톤 인공지능 예측 국내 버스"},
    {"사이트": "구글 데이터셋 검색", "분류": "경진대회·해외", "아이콘": "🔎", "주소": "https://datasetsearch.research.google.com",
     "검색": "https://datasetsearch.research.google.com/search?query={q}",
     "설명": "전 세계 데이터셋을 찾아 주는 Google 검색 엔진, 데이터 유형별 필터 가능",
     "예시": "covid-19, water quality 등", "제공형태": "원 사이트로 연결",
     "키워드": "검색 해외 영어 글로벌 데이터셋"},
    {"사이트": "UN Data", "분류": "경진대회·해외", "아이콘": "🌐", "주소": "https://data.un.org",
     "검색": None,
     "설명": "UN이 운영하는 공공 데이터 포털로 경제·인구·환경 등 국가별 정보 제공",
     "예시": "국가별 인구, SDG 지표, 무역 통계", "제공형태": "파일, API",
     "키워드": "국제 국가 세계 인구 경제 환경 SDG 영어"},
    {"사이트": "Books to Scrape", "분류": "웹 수집 연습", "아이콘": "📚", "주소": "https://books.toscrape.com",
     "검색": None,
     "설명": "웹 크롤링·스크래핑 연습을 위해 만든 교육용 사이트 (가격·평점은 무작위 값)",
     "예시": "도서명, 가격, 재고 상태 수집 실습", "제공형태": "HTML 웹페이지",
     "키워드": "크롤링 스크래핑 웹 연습 BeautifulSoup 도서"},
]

COLLECTION_METHODS = pd.DataFrame([
    {"수집 방법": "공개 데이터 활용", "언제 사용할까?": "신뢰도 있는 데이터를 활용해 분석(예측)하고자 할 때", "예": "공공데이터포털, KOSIS"},
    {"수집 방법": "웹 스크래핑", "언제 사용할까?": "웹 페이지 안의 특정 데이터를 수집하고자 할 때", "예": "상품명·가격 추출"},
    {"수집 방법": "설문 조사", "언제 사용할까?": "사람들의 의견을 직접 수집하고자 할 때", "예": "급식 메뉴 선호도"},
    {"수집 방법": "센서 데이터 수집", "언제 사용할까?": "사물 인터넷 기기로 직접 데이터를 수집하고자 할 때", "예": "교실 소음(dB) 측정"},
])


def site_matches(site, query):
    text = " ".join(str(site[k]) for k in ("사이트", "분류", "설명", "예시", "제공형태", "키워드")).lower()
    return all(word in text for word in query.lower().split())


# 학생들이 문제 정의에 자주 쓰는 표현 → 관련 사이트 (키워드 열에 없는 일상어를 보완)
QUESTION_HINTS = {
    "공공데이터포털": "버스 지하철 교통 자전거 주차 사고 미세먼지 대기 쓰레기 재활용 관광 여행 식당 가게 상권 학교 급식 병원 의료 도서관 공원",
    "국가통계포털 KOSIS": "인구 저출산 출산 고령화 노인 가구 1인가구 취업 일자리 실업 소득 물가 경제 결혼 사망 수명 범죄 학생수 교육비 사교육 학업 공부 학원",
    "KOSIS 100대 지표": "인구 경제 성장 물가 고용 추세 변화",
    "마이크로데이터 통합서비스 MDIS": "공부 학습 시험 성적 점수 학업 청소년 학생 설문 수면 스마트폰 게임 도박 흡연 음주 우울 스트레스 행복 건강 가계 사교육 용돈",
    "원윈도우": "급식 잔반 음식물 학교 데이터 통합",
    "통계지리정보서비스 SGIS": "지역 동네 우리동네 시군구 지도 위치 거리 상권 인구분포 주택 아파트",
    "기상자료개방포털": "날씨 기온 온도 더위 폭염 추위 한파 강수 장마 폭우 태풍 바람 풍속 습도 일조 기후 지구온난화 온난화 미세먼지 계절",
    "과수생육·품질관리시스템": "사과 배 과일 과수 농작물 농사 재배 수확 당도 기후변화 온난화",
    "농식품 빅데이터 거래소 KADX": "급식 잔반 음식 식품 농산물 소비 먹거리 식습관",
    "문화셈터 (문화체육관광 통계포털)": "비만 체력 운동 체육 키 몸무게 체중 BMI 건강 스포츠 여행 관광 문화 독서 영화 공연 여가",
    "캐글 Kaggle": "축구 야구 농구 스포츠 선수 영화 타이타닉 이미지 인공지능 머신러닝 예측 해외",
    "데이콘 DACON": "예측 인공지능 머신러닝 대회 버스 해커톤",
    "구글 데이터셋 검색": "해외 외국 세계 영어 논문 연구",
    "UN Data": "세계 국가별 나라 국제 해외 개발도상국 SDG 무역 환경 인구",
    "Books to Scrape": "크롤링 스크래핑 웹 책 도서 가격 수집 연습",
}

JOSA_SUFFIXES = sorted(
    ["으로부터", "에서는", "이라면", "일수록", "을수록", "할수록", "수록", "에서", "으로", "에게", "까지", "부터",
     "보다", "처럼", "이나", "이랑", "하고", "와", "과", "은", "는", "이", "가", "을", "를", "의", "에", "로", "도", "만"],
    key=len, reverse=True,
)
QUESTION_STOPWORDS = {"얼마나", "어떻게", "어떤", "무엇", "정말", "관계", "영향", "관련", "차이", "많은", "적은", "우리", "사람",
                      "가장", "많을", "적을", "높을", "낮을", "클수록", "작을"}


def extract_question_terms(question):
    # 문제 문장에서 조사·의문형 어미를 떼어 내고 검색에 쓸 만한 낱말만 남긴다.
    terms = []
    for token in re.findall(r"[가-힣A-Za-z0-9]+", question):
        for suffix in JOSA_SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 2:
                token = token[: -len(suffix)]
                break
        if len(token) < 2 or token in QUESTION_STOPWORDS or token[-1] in "까요다면고지" or token.endswith("수록"):
            continue
        if token not in terms:
            terms.append(token)
    return terms


def recommend_sites(question):
    q = question.lower()
    ranked = []
    for site in PUBLIC_DATA_SITES:
        words = f"{site['키워드']} {QUESTION_HINTS.get(site['사이트'], '')}".lower().split()
        hits = list(dict.fromkeys(w for w in words if len(w) >= 2 and w in q))
        if hits:
            ranked.append((len(hits), site, hits))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [(site, hits) for _, site, hits in ranked]


def render_question_recommendations(question):
    st.markdown("#### 📌 내 문제에 맞는 데이터 사이트")
    if not question:
        st.info("위 ① 문제 정의에 질문을 적으면, 그 질문과 관련된 공개 데이터 사이트를 추천해 드려요.")
        return

    recommendations = recommend_sites(question)
    matched = list(dict.fromkeys(w for _, hits in recommendations for w in hits))
    terms = list(dict.fromkeys(matched + extract_question_terms(question)))

    if recommendations:
        st.success(f"‘{question}’ 문제와 관련된 사이트 {len(recommendations)}곳을 찾았어요. 관련도 높은 순서입니다.")
        top = recommendations[:3]
        for col, (site, hits) in zip(st.columns(len(top)), top):
            with col.container(border=True):
                st.markdown(f"##### {site['아이콘']} {site['사이트']}")
                st.caption(f"추천 이유: {', '.join(hits)}")
                st.write(site["설명"])
                st.markdown(f"**예시** {site['예시']}")
                st.link_button("사이트 열기 ↗", site["주소"], width="stretch")
        if len(recommendations) > 3:
            with st.expander(f"관련 사이트 {len(recommendations) - 3}곳 더 보기"):
                for site, hits in recommendations[3:]:
                    st.markdown(f"- {site['아이콘']} [{site['사이트']}]({site['주소']}) — {site['예시']} "
                                f"<span style='color:gray'>(추천 이유: {', '.join(hits)})</span>",
                                unsafe_allow_html=True)
    else:
        st.warning("질문과 딱 맞는 사이트를 찾지 못했어요. 아래 핵심 낱말로 포털에서 직접 검색해 보세요.")

    if terms:
        term = st.pills(
            "핵심 낱말을 골라 포털에서 바로 검색하기",
            terms,
            default=terms[0],
            key=f"question_term_{abs(hash(question))}",
        )
        if term:
            search_sites = [s for s in PUBLIC_DATA_SITES if s["검색"]]
            for col, site in zip(st.columns(len(search_sites)), search_sites):
                col.link_button(
                    f"{site['아이콘']} {site['사이트']}에서 ‘{term}’",
                    site["검색"].format(q=urllib.parse.quote(term)),
                    width="stretch",
                )


def render_public_data_finder(question=""):
    render_question_recommendations(question.strip())

    # 눈에 잘 띄도록 key가 있는 컨테이너로 감싸 펼치기 막대를 꾸민다.
    with st.container(key="site_finder"):
        with st.expander(f"🔍 공개 데이터 사이트 {len(PUBLIC_DATA_SITES)}곳 한눈에 살펴보기 · 수집 방법 · 체크리스트  👉 눌러서 펼치기"):
            st.caption("문제에 맞는 데이터를 어디서 구할 수 있는지 찾아보고, 버튼을 눌러 사이트로 바로 이동해 보세요.")

            categories = list(dict.fromkeys(s["분류"] for s in PUBLIC_DATA_SITES))
            m1, m2, m3 = st.columns(3)
            m1.metric("소개된 사이트", f"{len(PUBLIC_DATA_SITES)}곳")
            m2.metric("분야", f"{len(categories)}개")
            m3.metric("키워드로 바로 검색", f"{sum(1 for s in PUBLIC_DATA_SITES if s['검색'])}곳")

            find_tab, table_tab, method_tab, check_tab = st.tabs(
                ["🔎 사이트 찾기", "📋 한눈에 보기", "🧭 수집 방법 고르기", "✅ 수집 전 체크리스트"]
            )

            with find_tab:
                query = st.text_input(
                    "찾고 싶은 데이터 주제를 입력하세요",
                    placeholder="예) 기온, 인구, 급식, 비만, 사과, 축구",
                    key="site_query",
                ).strip()
                picked = st.pills("분야 선택", categories, selection_mode="multi", key="site_category")

                results = [
                    s for s in PUBLIC_DATA_SITES
                    if (not picked or s["분류"] in picked) and (not query or site_matches(s, query))
                ]

                if query:
                    search_sites = [s for s in PUBLIC_DATA_SITES if s["검색"]]
                    st.markdown(f"**‘{query}’ 사이트에서 바로 검색하기**")
                    for col, site in zip(st.columns(len(search_sites)), search_sites):
                        col.link_button(
                            f"{site['아이콘']} {site['사이트']}",
                            site["검색"].format(q=urllib.parse.quote(query)),
                            width="stretch",
                        )

                st.markdown(f"**추천 사이트 {len(results)}곳**")
                if not results:
                    st.info("검색어와 딱 맞는 사이트가 없어요. 위의 ‘바로 검색하기’ 버튼으로 포털에서 직접 찾아보세요.")
                for row_start in range(0, len(results), 3):
                    for col, site in zip(st.columns(3), results[row_start:row_start + 3]):
                        with col.container(border=True):
                            st.markdown(f"##### {site['아이콘']} {site['사이트']}")
                            st.caption(f"{site['분류']} · {site['제공형태']}")
                            st.write(site["설명"])
                            st.markdown(f"**예시** {site['예시']}")
                            st.link_button("사이트 열기 ↗", site["주소"], width="stretch")

            with table_tab:
                overview = pd.DataFrame(PUBLIC_DATA_SITES)[["분류", "사이트", "설명", "예시", "제공형태", "주소"]]
                st.dataframe(
                    overview,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "주소": st.column_config.LinkColumn("바로가기", display_text="열기 ↗"),
                        "설명": st.column_config.TextColumn("어떤 데이터를 얻을 수 있나요?", width="large"),
                        "예시": st.column_config.TextColumn("예시", width="medium"),
                    },
                )
                st.download_button(
                    "사이트 목록 CSV로 내려받기",
                    overview.to_csv(index=False).encode("utf-8-sig"),
                    file_name="공개데이터_사이트목록.csv",
                    mime="text/csv",
                )

            with method_tab:
                st.dataframe(COLLECTION_METHODS, hide_index=True, width="stretch")
                st.markdown(
                    '<div class="note">공개 데이터에서 원하는 데이터를 구할 수 없거나 특별한 조건에 맞는 데이터가 필요하면 '
                    '<b>관찰·측정·설문 조사</b>로 직접 수집합니다. 설문의 질문과 선택지는 구체적·명확하고, '
                    '선택지는 서로 배타적이어야 합니다.</div>',
                    unsafe_allow_html=True,
                )
                with st.popover("웹 크롤링 vs 웹 스크래핑"):
                    st.markdown(
                        "| 구분 | 웹 크롤링 | 웹 스크래핑 |\n|---|---|---|\n"
                        "| 목적 | 웹페이지 탐색 및 수집 | 웹페이지 내 데이터 추출 |\n"
                        "| 행동 | 링크를 따라가며 페이지 수집 | 페이지 안에서 필요한 데이터 추출 |\n"
                        "| 비유 | 도서관에서 책을 찾는 일 | 책에서 필요한 내용을 발췌하는 일 |"
                    )
                    st.caption("실습은 robots.txt를 확인하고, Books to Scrape 같은 연습용 사이트나 공공 API로 진행합니다.")

            with check_tab:
                st.caption("데이터를 내려받기 전에 하나씩 확인해 보세요.")
                checks = [
                    "수집 가능한 데이터인가?",
                    "정보 보호에 저촉되지 않는 데이터인가? (개인정보·민감정보 제외)",
                    "신뢰할 수 있는 데이터인가? (출처가 정부·대학·공공기관 등으로 명확한가)",
                    "데이터를 수집하는 데 비용 문제가 발생하지 않는가?",
                    "데이터 이용 허락 범위와 저작권·출처 표기 방법을 확인했는가?",
                    "메타 데이터(수집 주체, 속성의 의미, 수집 방법)를 확인했는가?",
                    "특정 지역이나 집단에 치우치지 않았는가? (선택 편향)",
                ]
                done = sum(st.checkbox(item, key=f"collect_check_{i}") for i, item in enumerate(checks))
                st.progress(done / len(checks), text=f"{done}/{len(checks)} 확인 완료")
                if done == len(checks):
                    st.success("수집 준비 완료! 내려받은 CSV를 아래 ‘② 데이터 수집 방법’에서 업로드해 보세요.")


STEPS = ["01. 데이터 수집", "02. 수집한 데이터의 특성", "03. 데이터 전처리", "04. 데이터 시각화"]
NO_DATA_MSG = "먼저 '02. 수집한 데이터의 특성' 탭에서 CSV 파일을 업로드하거나 예시 데이터를 선택해 주세요."
STAT_CONCEPTS = {
    "평균": "데이터의 총합을 데이터의 전체 개수로 나눈 값",
    "분산": "평균과 각 데이터의 차이(편차)를 제곱해서 더한 값을 데이터의 전체 개수로 나눈 값",
    "표준편차": "분산의 제곱근으로, 데이터가 평균에서 얼마나 퍼져 있는지 나타내는 척도",
    "최솟값": "데이터 중 가장 작은 값",
    "중앙값": "데이터를 크기순으로 늘어놓았을 때 중앙에 위치하는 값",
    "최댓값": "데이터 중 가장 큰 값",
}


def step_header(title, desc):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subbox">{desc}</div>', unsafe_allow_html=True)


def next_step(index):
    if index < len(STEPS):
        st.caption(f"다음 단계 ▶ 위쪽의 **{STEPS[index]}** 탭을 눌러 이어서 실습하세요.")


def numeric_stats(df, cols):
    # 교과서 정의대로 분산·표준편차는 전체 개수(N)로 나눈다.
    d = df[cols]
    return pd.DataFrame({
        "평균": d.mean(), "분산": d.var(ddof=0), "표준편차": d.std(ddof=0),
        "최솟값": d.min(), "중앙값": d.median(), "최댓값": d.max(),
    }).round(2)


def outlier_count(series):
    s = series.dropna()
    if s.empty:
        return 0
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int((~s.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)).sum())


# ── 💻 코드 실습(노트북) ─────────────────────────────────────────────
# 학생 코드는 별도 프로세스에서 실행한다. 셀 사이의 변수를 이어 쓰기 위해
# 셀을 실행할 때마다 1번 셀부터 그 셀까지 차례로 다시 실행한다.
NOTEBOOK_TIMEOUT = 15
NOTEBOOK_RUNNER = r'''
import ast, base64, builtins, contextlib, io, json, sys, traceback

cells = json.loads(sys.stdin.read())
result_path = sys.argv[1]
ALLOWED = {"pandas", "numpy", "math", "statistics", "random", "datetime", "collections",
           "itertools", "re", "json", "matplotlib", "sklearn", "seaborn", "scipy"}
BLOCKED_ATTRS = {"os", "sys", "subprocess", "system", "popen", "builtins", "importlib",
                 "shutil", "socket", "eval", "exec", "compile"}

try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 ** 3, 2 * 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (50 * 1024 ** 2, 50 * 1024 ** 2))
except Exception:
    pass

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in ALLOWED:
        raise ImportError(f"이 실습 환경에서는 '{name}' 모듈을 불러올 수 없습니다. "
                          f"사용 가능: {', '.join(sorted(ALLOWED))}")
    return real_import(name, globals, locals, fromlist, level)

safe_builtins = dict(vars(builtins))
for name in ("open", "exec", "eval", "compile", "input", "breakpoint", "globals", "locals",
             "vars", "getattr", "setattr", "delattr", "exit", "quit", "help", "memoryview"):
    safe_builtins.pop(name, None)
safe_builtins["__import__"] = guarded_import
namespace = {"__builtins__": safe_builtins, "__name__": "__main__"}

def check(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [n.name for n in node.names]
            for name in names:
                guarded_import(name, level=getattr(node, "level", 0))
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError(f"내부 전용 이름 '{node.id}'은(는) 사용할 수 없습니다.")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in BLOCKED_ATTRS):
            raise ValueError(f"'.{node.attr}' 속성은 이 실습 환경에서 사용할 수 없습니다.")

def render(value):
    pd = sys.modules.get("pandas")
    if pd is not None and isinstance(value, (pd.DataFrame, pd.Series)):
        frame = value.to_frame() if isinstance(value, pd.Series) else value
        return {"kind": "table", "shape": list(frame.shape),
                "data": frame.head(500).to_json(orient="split", date_format="iso", default_handler=str)}
    text = repr(value)
    return {"kind": "text", "text": text[:20000]}

def figures():
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return []
    images = []
    for num in plt.get_fignums():
        buffer = io.BytesIO()
        plt.figure(num).savefig(buffer, format="png", bbox_inches="tight", dpi=110)
        images.append(base64.b64encode(buffer.getvalue()).decode())
    plt.close("all")
    return images

results = []
for index, source in enumerate(cells, start=1):
    out = io.StringIO()
    item = {"stdout": "", "value": None, "images": [], "error": ""}
    filename = f"셀{index}"
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            tree = ast.parse(source, filename=filename)
            check(tree)
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            exec(compile(tree, filename, "exec"), namespace)
            if last is not None:
                value = eval(compile(ast.Expression(last.value), filename, "eval"), namespace)
                if value is not None:
                    item["value"] = render(value)
            item["images"] = figures()
    except BaseException as error:
        line = getattr(error, "lineno", None) if isinstance(error, SyntaxError) else None
        for frame in traceback.extract_tb(error.__traceback__):
            if frame.filename == filename:
                line = frame.lineno
        item["error"] = f"{type(error).__name__}: {error}" + (f"  (줄 {line})" if line else "")
    item["stdout"] = out.getvalue()[:20000]
    results.append(item)
    if item["error"]:
        break

with open(result_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False)
'''


def run_notebook_cells(sources, files):
    with tempfile.TemporaryDirectory() as workdir, tempfile.TemporaryDirectory() as resultdir:
        for name, data in files.items():
            with open(os.path.join(workdir, name), "wb") as f:
                f.write(data)
        before = {name: os.path.getmtime(os.path.join(workdir, name)) for name in os.listdir(workdir)}
        result_path = os.path.join(resultdir, "result.json")
        # HOME은 그대로 두어야 사용자 영역(~/.local)에 설치된 pandas 등을 찾을 수 있다.
        env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", workdir),
               "MPLBACKEND": "Agg", "MPLCONFIGDIR": resultdir, "PYTHONIOENCODING": "utf-8"}
        try:
            proc = subprocess.run(
                [sys.executable, "-c", NOTEBOOK_RUNNER, result_path],
                input=json.dumps(sources), capture_output=True, text=True,
                cwd=workdir, env=env, timeout=NOTEBOOK_TIMEOUT, check=False,
            )
        except subprocess.TimeoutExpired:
            return None, {}, f"실행 시간이 {NOTEBOOK_TIMEOUT}초를 넘어 중단했습니다. 무한 반복이 있는지 확인해 보세요."
        if not os.path.exists(result_path):
            return None, {}, "코드 실행이 비정상적으로 끝났습니다. " + proc.stderr.strip()[-500:]
        with open(result_path, encoding="utf-8") as f:
            results = json.load(f)
        # 학생 코드가 to_csv 등으로 새로 만들거나 고친 파일은 파일 목록에 다시 담는다.
        created = {}
        for name in os.listdir(workdir):
            path = os.path.join(workdir, name)
            if os.path.isfile(path) and before.get(name) != os.path.getmtime(path) and os.path.getsize(path) <= 20 * 1024 ** 2:
                with open(path, "rb") as f:
                    created[name] = f.read()
        return results, created, ""


def show_cell_output(output):
    if output.get("stdout"):
        st.code(output["stdout"].rstrip("\n"), language="text")
    value = output.get("value")
    if value and value["kind"] == "table":
        frame = pd.read_json(io.StringIO(value["data"]), orient="split")
        rows, cols = value["shape"]
        st.dataframe(frame, width="stretch")
        st.caption(f"{rows:,} rows × {cols} columns" + (" (앞 500행만 표시)" if rows > 500 else ""))
    elif value:
        st.code(value["text"], language="text")
    for image in output.get("images", []):
        st.image(base64.b64decode(image))
    if output.get("error"):
        st.error(output["error"])


def notebook_default_code(file_name):
    return (f"import pandas as pd\ndata = pd.read_csv('{file_name}')\ndata.info()\ndata.describe()"
            if file_name else "import pandas as pd\n# 왼쪽 파일 패널에 CSV를 업로드한 뒤 파일 이름을 적어 보세요.\n")


def notebook_state():
    ss = st.session_state
    ss.setdefault("nb_files", {})
    ss.setdefault("nb_cells", [{"id": 0, "count": None, "output": None}])
    ss.setdefault("nb_next_id", 1)
    ss.setdefault("nb_counter", 0)
    return ss


def notebook_run_upto(position):
    ss = st.session_state
    cells = ss["nb_cells"][: position + 1]
    sources = [ss.get(f"nb_code_{c['id']}", "") for c in cells]
    results, created, error = run_notebook_cells(sources, ss["nb_files"])
    ss["nb_files"].update(created)
    if error:
        cells[-1]["output"] = {"error": error}
        return
    for cell, output in zip(cells, results):
        ss["nb_counter"] += 1
        cell["count"] = ss["nb_counter"]
        cell["output"] = output
    for cell in cells[len(results):]:
        cell["count"], cell["output"] = None, None


def notebook_add_cell(code=""):
    ss = st.session_state
    ss[f"nb_code_{ss['nb_next_id']}"] = code
    ss["nb_cells"].append({"id": ss["nb_next_id"], "count": None, "output": None})
    ss["nb_next_id"] += 1


def notebook_delete_cell(position):
    st.session_state["nb_cells"].pop(position)
    if not st.session_state["nb_cells"]:
        notebook_add_cell()


def notebook_reset():
    for key in [k for k in st.session_state if str(k).startswith("nb_")]:
        del st.session_state[key]


raw_df = None
data_id = None
clean_df = None

def wkey(name):
    # 데이터가 바뀌면 위젯 선택값도 새로 시작하도록 키에 데이터 식별자를 붙인다.
    return f"{name}_{data_id}"


ANALYSIS_MENUS = ["🔍 탐색적 데이터 분석", "📈 회귀 분석", "🧩 군집 분석", "🛒 연관 분석", "💻 코드 실습"]
# 상단 메뉴만 글자를 키우기 위해 key가 있는 컨테이너 안에 탭을 만든다.
with st.container(key="step_menu"):
    step1, step2, step3, step4, eda_tab, reg_tab, cluster_tab, assoc_tab, code_tab = st.tabs(
        [f"{'①②③④'[i]} {name}" for i, name in enumerate(STEPS)] + ANALYSIS_MENUS
    )

with step1:
    step_header("01. 데이터 수집", "문제를 정의하고, 문제 해결에 필요한 데이터를 어디서 어떻게 구할지 찾아 내려받습니다.")
    research_question = st.text_input(
        "① 문제 정의: 이 데이터로 알아보고 싶은 질문을 적어 보세요",
        placeholder="예) 공부시간이 많을수록 시험점수가 높을까?",
    )

    render_public_data_finder(research_question)

    st.markdown('<div class="note">데이터를 내려받았다면 <b>02. 수집한 데이터의 특성</b> 탭에서 CSV 파일을 업로드해 데이터의 특성을 살펴보세요.</div>', unsafe_allow_html=True)
    next_step(1)

with step2:
    step_header("02. 수집한 데이터의 특성", "수집한 CSV 파일을 업로드하고, 데이터의 크기·열의 종류·통계값·품질을 살펴보며 내 질문을 해결할 수 있는 데이터인지 판단합니다.")
    with st.container(border=True):
        st.markdown("#### ① 수집한 데이터 불러오기")
        source = st.radio("데이터 불러오기 방법", ["CSV 파일 업로드", "예시 데이터 사용"], horizontal=True)
        if source == "CSV 파일 업로드":
            collected_file = st.file_uploader("수집한 CSV 파일을 업로드하세요", type=["csv"], key="collect_csv")
            if collected_file is not None:
                try:
                    raw_df = read_uploaded_csv(collected_file)
                    data_id = f"{collected_file.name}_{collected_file.size}"
                except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
                    st.error(f"CSV 파일을 읽을 수 없습니다. {error}")
        else:
            sample_name = st.selectbox("예시 데이터 선택", list(SAMPLE_DATASETS))
            raw_df = SAMPLE_DATASETS[sample_name]()
            data_id = sample_name
        if raw_df is not None:
            st.success(f"데이터 {len(raw_df):,}행 × {len(raw_df.columns)}열을 불러왔습니다.")

    if raw_df is None:
        st.info("CSV 파일을 업로드하면 데이터의 특성이 나타나고, 03~04단계 실습이 이 데이터로 진행됩니다.")
    else:
        st.markdown("#### ② 데이터 구조 살펴보기")
        raw_num_cols, raw_cat_cols = split_columns(raw_df)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("행 개수", f"{len(raw_df):,}")
        m2.metric("열 개수", len(raw_df.columns))
        m3.metric("결측값 수", f"{int(raw_df.isna().sum().sum()):,}")
        m4.metric("중복 행 수", int(raw_df.duplicated().sum()))
        st.dataframe(raw_df.head(10), width="stretch")

        left, right = st.columns([1.3, 0.7])
        with left:
            st.dataframe(column_profile(raw_df), hide_index=True, width="stretch")
        with right:
            st.write(f"- 수치형 열 {len(raw_num_cols)}개: {', '.join(map(str, raw_num_cols)) or '없음'}")
            st.write(f"- 범주형 열 {len(raw_cat_cols)}개: {', '.join(map(str, raw_cat_cols)) or '없음'}")
            missing_cols = [str(c) for c in raw_df.columns if raw_df[c].isna().any()]
            if missing_cols:
                st.write(f"- 결측값이 있는 열: {', '.join(missing_cols)} → 03. 데이터 전처리에서 처리")
            time_cols = [str(c) for c in raw_df.columns if re.search(r"연도|년|날짜|year|date", str(c), re.IGNORECASE)]
            if time_cols:
                st.write(f"- 시간 관련 열({', '.join(time_cols)})이 있어 시계열 분석이 가능합니다.")
            st.markdown("<div class='note'>데이터를 수집한 뒤에는 '이 데이터로 내 질문을 해결할 수 있는가?'를 판단하는 과정이 꼭 필요합니다.</div>", unsafe_allow_html=True)

        st.markdown("#### ③ 통계로 특성 파악하기")
        if not raw_num_cols:
            st.info("수치형 열이 없어 통계값을 계산할 수 없습니다. 03. 데이터 전처리에서 숫자로 변환해 보세요.")
        else:
            st.dataframe(numeric_stats(raw_df, raw_num_cols), width="stretch")
            stat_col = st.selectbox("자세히 볼 열", raw_num_cols, key=wkey("trait_col"))
            col_stats = numeric_stats(raw_df, [stat_col]).iloc[0]
            for col, name in zip(st.columns(6), STAT_CONCEPTS):
                col.metric(name, f"{col_stats[name]:,.2f}", help=STAT_CONCEPTS[name])
            if col_stats["평균"] != col_stats["중앙값"]:
                gap = "크므로 큰 값 쪽으로 치우친" if col_stats["평균"] > col_stats["중앙값"] else "작으므로 작은 값 쪽으로 치우친"
                st.write(f"- {stat_col}의 평균이 중앙값보다 {gap} 분포일 수 있습니다.")
        with st.expander("📖 통계 개념 정리 (평균·분산·표준편차·최솟값·중앙값·최댓값)"):
            st.table(pd.DataFrame({"개념": list(STAT_CONCEPTS), "설명": list(STAT_CONCEPTS.values())}).set_index("개념"))

        st.markdown("#### ④ 데이터 품질 점검")
        quality = pd.DataFrame({
            "결측값": raw_df.isna().sum(),
            "이상치(IQR)": [outlier_count(raw_df[c]) if c in raw_num_cols else 0 for c in raw_df.columns],
        })
        q1, q2, q3 = st.columns(3)
        q1.metric("누락(결측값)", f"{int(quality['결측값'].sum()):,}")
        q2.metric("중복 행", int(raw_df.duplicated().sum()))
        q3.metric("이상치 의심 값", f"{int(quality['이상치(IQR)'].sum()):,}")
        problems = quality[(quality["결측값"] > 0) | (quality["이상치(IQR)"] > 0)]
        if problems.empty and not raw_df.duplicated().any():
            st.success("오류·누락·중복이 발견되지 않았습니다. 그래도 특정 지역이나 집단에 치우친 데이터는 아닌지 확인해 보세요.")
        else:
            st.dataframe(problems, width="stretch")
            st.warning("발견된 결측값·중복·이상치는 03. 데이터 전처리 단계에서 처리합니다.")

    next_step(2)

with step3:
    step_header("03. 데이터 전처리", "분석에 필요한 열을 고르고, 결측값·중복·이상치를 처리해 분석하기 좋은 데이터로 정리합니다.")
    if raw_df is None:
        st.info(NO_DATA_MSG)
    else:
        p1, p2 = st.columns(2)
        with p1:
            default_cols = [c for c in raw_df.columns if raw_df[c].notna().any()]
            selected_cols = st.multiselect("① 분석에 사용할 열 선택", raw_df.columns.tolist(), default=default_cols, key=wkey("prep_cols"))
            convert_cols = st.multiselect(
                "② 숫자로 변환할 열 (숫자인데 문자로 읽힌 열)",
                [c for c in selected_cols if not is_numeric(raw_df[c])],
                key=wkey("prep_convert"),
            )
            drop_duplicates = st.checkbox("③ 중복 행 제거", value=True, key=wkey("prep_dup"))
        with p2:
            missing_method = st.selectbox(
                "④ 결측값 처리",
                ["그대로 두기", "결측값이 있는 행 삭제", "평균으로 채우기", "중앙값으로 채우기", "최빈값으로 채우기"],
                index=1,
                key=wkey("prep_missing"),
            )
            outlier_candidates = [c for c in selected_cols if c in convert_cols or is_numeric(raw_df[c])]
            outlier_cols = st.multiselect("⑤ 이상치 제거 (IQR 방법)", outlier_candidates, key=wkey("prep_outlier"))
            filter_candidates = [c for c in selected_cols if c not in outlier_candidates]
            filter_col = st.selectbox("⑥ 행 필터 기준 열 (선택)", ["사용 안 함", *filter_candidates], key=wkey("prep_filter_col"))
            filter_values = []
            if filter_col != "사용 안 함":
                value_options = raw_df[filter_col].dropna().astype(str).unique().tolist()
                filter_values = st.multiselect(f"'{filter_col}'에서 남길 값", value_options, key=wkey("prep_filter_val"))

        clean_df = raw_df[selected_cols].copy()
        for col in convert_cols:
            clean_df[col] = pd.to_numeric(clean_df[col].astype(str).str.replace(",", ""), errors="coerce")
        if drop_duplicates:
            clean_df = clean_df.drop_duplicates()

        num_cols, cat_cols = split_columns(clean_df)
        if missing_method == "결측값이 있는 행 삭제":
            clean_df = clean_df.dropna()
        elif missing_method != "그대로 두기":
            for col in clean_df.columns:
                if col in num_cols and missing_method == "평균으로 채우기":
                    fill_value = clean_df[col].mean()
                elif col in num_cols and missing_method == "중앙값으로 채우기":
                    fill_value = clean_df[col].median()
                else:
                    modes = clean_df[col].mode()
                    fill_value = modes.iloc[0] if not modes.empty else None
                if fill_value is not None:
                    clean_df[col] = clean_df[col].fillna(fill_value)

        for col in outlier_cols:
            q1, q3 = clean_df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            keep = clean_df[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr) | clean_df[col].isna()
            clean_df = clean_df[keep]

        if filter_values:
            clean_df = clean_df[clean_df[filter_col].astype(str).isin(filter_values)]
        clean_df = clean_df.reset_index(drop=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("전처리 후 행 수", f"{len(clean_df):,}", delta=f"{len(clean_df) - len(raw_df):,}")
        c2.metric("전처리 후 열 수", len(clean_df.columns), delta=len(clean_df.columns) - len(raw_df.columns))
        remaining_missing = int(clean_df.isna().sum().sum())
        c3.metric("남은 결측값", remaining_missing, delta=remaining_missing - int(raw_df.isna().sum().sum()), delta_color="inverse")
        st.dataframe(clean_df.head(10), width="stretch")

        missing_compare = pd.DataFrame({
            "전처리 전": raw_df[selected_cols].isna().sum(),
            "전처리 후": clean_df.isna().sum(),
        }).fillna(0)
        if missing_compare.to_numpy().sum() > 0:
            st.caption("열별 결측값 수 비교")
            st.bar_chart(missing_compare, stack=False)

        st.download_button(
            "전처리된 데이터 CSV 다운로드",
            data=clean_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
    next_step(3)

with step4:
    step_header("04. 데이터 시각화", "전처리한 데이터를 그래프로 나타내 분포·추세·관계를 한눈에 파악합니다.")
    if clean_df is None:
        st.info(NO_DATA_MSG)
    else:
        st.markdown("#### ① 그래프로 나타내기")
        num_cols, cat_cols = split_columns(clean_df)
        if clean_df.empty:
            st.warning("전처리 후 남은 데이터가 없습니다. 전처리 조건을 조정해 보세요.")
        elif not num_cols:
            st.warning("시각화하려면 수치형 열이 하나 이상 필요합니다. 03. 데이터 전처리의 ② 숫자로 변환할 열을 확인해 보세요.")
        else:
            color_options = ["없음", *[c for c in cat_cols if clean_df[c].nunique() <= 20]]
            vis_type = st.selectbox("그래프 유형", ["히스토그램", "막대형", "꺾은선형", "산점도", "상자그림"], key=wkey("vis_type"))
            v1, v2, v3 = st.columns(3)
            if vis_type == "히스토그램":
                x_col = v1.selectbox("값(수치형)", num_cols, key=wkey("vis_hist_x"))
                bins = v2.slider("구간 수", 5, 50, 15, key=wkey("vis_bins"))
                chart = alt.Chart(clean_df).mark_bar().encode(
                    x=alt.X(fld(x_col), bin=alt.Bin(maxbins=bins), title=str(x_col)),
                    y=alt.Y("count()", title="빈도"),
                )
            elif vis_type == "막대형":
                if not cat_cols:
                    st.info("막대형 그래프는 범주형 열이 필요합니다. 다른 그래프를 선택해 보세요.")
                    chart = None
                else:
                    x_col = v1.selectbox("범주(가로축)", cat_cols, key=wkey("vis_bar_x"))
                    y_col = v2.selectbox("값(세로축)", num_cols, key=wkey("vis_bar_y"))
                    agg = v3.selectbox("집계 방법", ["평균", "합계", "개수"], key=wkey("vis_bar_agg"))
                    agg_df = clean_df.groupby(x_col, dropna=True)[y_col].agg({"평균": "mean", "합계": "sum", "개수": "count"}[agg]).reset_index()
                    if len(agg_df) > 30:
                        st.caption("범주가 많아 값이 큰 상위 30개만 표시합니다.")
                        agg_df = agg_df.nlargest(30, y_col)
                    chart = alt.Chart(agg_df).mark_bar().encode(
                        x=alt.X(fld(x_col), sort="-y", title=str(x_col)),
                        y=alt.Y(fld(y_col), title=f"{y_col} ({agg})"),
                        tooltip=[fld(x_col), fld(y_col)],
                    )
            elif vis_type == "꺾은선형":
                x_col = v1.selectbox("가로축 (연도·순서)", clean_df.columns.tolist(), key=wkey("vis_line_x"))
                y_col = v2.selectbox("값(세로축)", [c for c in num_cols if c != x_col] or num_cols, key=wkey("vis_line_y"))
                color_col = v3.selectbox("색상 구분", color_options, key=wkey("vis_line_c"))
                group_cols = [x_col] if color_col == "없음" else [x_col, color_col]
                line_df = clean_df.groupby(group_cols, dropna=True)[y_col].mean().reset_index()
                chart = alt.Chart(line_df).mark_line(point=True).encode(
                    x=alt.X(fld(x_col), title=str(x_col)),
                    y=alt.Y(fld(y_col), title=f"{y_col} (평균)"),
                    tooltip=[fld(c) for c in group_cols] + [fld(y_col)],
                )
                if color_col != "없음":
                    chart = chart.encode(color=alt.Color(fld(color_col), title=str(color_col)))
                if color_col == "없음" and any(clean_df[c].nunique() > 20 for c in cat_cols):
                    st.caption("범주가 많은 열은 03. 데이터 전처리의 ⑥ 행 필터로 몇 개 값만 남기면 색상 구분에 사용할 수 있습니다.")
            elif vis_type == "산점도":
                x_col = v1.selectbox("가로축(X)", num_cols, key=wkey("vis_sc_x"))
                y_col = v2.selectbox("세로축(Y)", num_cols, index=min(1, len(num_cols) - 1), key=wkey("vis_sc_y"))
                color_col = v3.selectbox("색상 구분", color_options, key=wkey("vis_sc_c"))
                chart = alt.Chart(clean_df).mark_circle(size=70, opacity=0.7).encode(
                    x=alt.X(fld(x_col), title=str(x_col), scale=alt.Scale(zero=False)),
                    y=alt.Y(fld(y_col), title=str(y_col), scale=alt.Scale(zero=False)),
                    tooltip=[fld(c) for c in clean_df.columns[:6]],
                )
                if color_col != "없음":
                    chart = chart.encode(color=alt.Color(fld(color_col), title=str(color_col)))
            else:
                y_col = v1.selectbox("값(수치형)", num_cols, key=wkey("vis_box_y"))
                group_col = v2.selectbox("그룹(선택)", color_options, key=wkey("vis_box_g"))
                if group_col == "없음":
                    chart = alt.Chart(clean_df).mark_boxplot(size=60).encode(y=alt.Y(fld(y_col), title=str(y_col)))
                else:
                    chart = alt.Chart(clean_df).mark_boxplot().encode(
                        x=alt.X(fld(group_col), title=str(group_col)),
                        y=alt.Y(fld(y_col), title=str(y_col)),
                        color=alt.Color(fld(group_col), legend=None),
                    )
            if chart is not None:
                st.altair_chart(chart.properties(height=360), width="stretch")

    st.markdown("#### ② 텍스트 데이터 시각화: 워드클라우드")
    with st.form("wordcloud_form"):
        input_text = st.text_area(
            "분석할 텍스트를 입력하세요",
            value="데이터 분석은 데이터를 수집하고 정리하여 의미 있는 정보를 찾는 과정입니다. "
            "데이터 시각화는 복잡한 데이터를 쉽게 이해하도록 도와줍니다. "
            "좋은 분석은 정확한 데이터와 적절한 질문에서 시작합니다.",
            height=120,
        )
        start_analysis = st.form_submit_button("시작", type="primary")

    if start_analysis:
        st.session_state["wordcloud_input_text"] = input_text

    uploaded_file = st.file_uploader(
        "텍스트 파일 또는 CSV 파일을 첨부하면 자동으로 분석합니다",
        type=["txt", "md", "csv"],
        key="wordcloud_file",
    )

    analysis_text = st.session_state.get("wordcloud_input_text")
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_text = None
        for encoding in ("utf-8-sig", "cp949"):
            try:
                file_text = file_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if file_text is None:
            st.error("파일 인코딩을 읽을 수 없습니다. UTF-8 또는 CP949 파일을 사용해 주세요.")
            analysis_text = None
        elif uploaded_file.name.lower().endswith(".csv"):
            try:
                csv_data = pd.read_csv(io.StringIO(file_text), header=None, dtype=str, keep_default_na=False)
                analysis_text = " ".join(csv_data.values.flatten().tolist())
                st.success(f"{uploaded_file.name} 파일 분석 완료")
            except (pd.errors.ParserError, pd.errors.EmptyDataError):
                st.error("CSV 파일 형식을 확인해 주세요.")
                analysis_text = None
        else:
            analysis_text = file_text
            st.success(f"{uploaded_file.name} 파일 분석 완료")

    if analysis_text is None:
        st.info("텍스트를 입력하고 시작을 누르거나 파일을 첨부해 주세요.")
    else:
        words = re.findall(r"[가-힣A-Za-z0-9]+", analysis_text)
        word_counts = Counter(word for word in words if len(word) > 1)

    if analysis_text is not None and word_counts:
        max_count = max(word_counts.values())
        cloud_colors = ["#1d4d3b", "#327a66", "#4f8290", "#b66b3d", "#6a7e45"]
        cloud_words = []
        for index, (word, count) in enumerate(word_counts.most_common(60)):
            font_size = 18 + (count - 1) / max(max_count - 1, 1) * 34
            cloud_words.append(
                f'<span style="font-size:{font_size:.0f}px;color:{cloud_colors[index % len(cloud_colors)]};'
                f'font-weight:{700 if count == max_count else 500};padding:5px 9px">'
                f'{html.escape(word)}<small style="font-size:12px;padding-left:4px">{count}</small></span>'
            )
        st.markdown(
            '<div style="min-height:240px;display:flex;flex-wrap:wrap;align-items:center;'
            'justify-content:center;align-content:center;gap:5px 8px;padding:24px 12px;'
            'background:rgba(141,179,107,0.08);border-radius:8px">'
            + "".join(cloud_words)
            + "</div>",
            unsafe_allow_html=True,
        )
        frequency_df = pd.DataFrame(word_counts.most_common(10), columns=["단어", "빈도"])
        st.dataframe(frequency_df, hide_index=True, width="stretch")
    elif analysis_text is not None:
        st.info("두 글자 이상의 단어를 입력하면 워드클라우드가 표시됩니다.")

with eda_tab:
    step_header("탐색적 데이터 분석(EDA)", "04단계까지 마친 데이터로 탐색적 데이터 분석을 이어서 해 보세요.")
    if clean_df is None or clean_df.empty:
        st.info("01~03단계(수집·특성 파악·전처리)를 마치면 전처리된 데이터로 EDA를 진행합니다.")
    else:
        eda_num_cols, eda_cat_cols = split_columns(clean_df)
        if research_question:
            st.markdown(f"<div class='note'>🔎 탐구 질문: {html.escape(research_question)}</div>", unsafe_allow_html=True)
        st.caption(f"03. 데이터 전처리를 거친 데이터 {len(clean_df):,}행 × {len(clean_df.columns)}열을 분석합니다.")

        st.markdown("#### ① 기술 통계")
        if eda_num_cols:
            st.dataframe(clean_df[eda_num_cols].describe().T.round(2), width="stretch")
        if eda_cat_cols:
            st.dataframe(clean_df[eda_cat_cols].describe().T, width="stretch")

        st.markdown("#### ② 변수 하나씩 살펴보기 (단변량 분석)")
        uni_col = st.selectbox("살펴볼 열", clean_df.columns.tolist(), key=wkey("eda_uni"))
        if uni_col in eda_num_cols:
            s = clean_df[uni_col].dropna()
            u1, u2, u3, u4 = st.columns(4)
            u1.metric("평균", f"{s.mean():,.2f}")
            u2.metric("중앙값", f"{s.median():,.2f}")
            u3.metric("표준편차", f"{s.std():,.2f}")
            u4.metric("왜도", f"{s.skew():.2f}")
            hist = alt.Chart(clean_df).mark_bar().encode(
                x=alt.X(fld(uni_col), bin=alt.Bin(maxbins=20), title=str(uni_col)),
                y=alt.Y("count()", title="빈도"),
            ).properties(height=260)
            box = alt.Chart(clean_df).mark_boxplot(size=40).encode(x=alt.X(fld(uni_col), title=str(uni_col))).properties(height=80)
            st.altair_chart(alt.vconcat(hist, box), width="stretch")
            skew = s.skew()
            if abs(skew) < 0.5:
                st.write("- 분포가 대체로 좌우 대칭입니다.")
            else:
                st.write(f"- 분포가 {'오른쪽' if skew > 0 else '왼쪽'}으로 꼬리가 긴 형태입니다. 평균보다 중앙값이 대푯값으로 적절할 수 있습니다.")
        else:
            counts = clean_df[uni_col].astype(str).value_counts().head(20).rename_axis(str(uni_col)).reset_index(name="빈도")
            st.altair_chart(
                alt.Chart(counts).mark_bar().encode(
                    x=alt.X("빈도:Q"), y=alt.Y(f"{fld(uni_col)}:N", sort="-x", title=str(uni_col)),
                ).properties(height=max(160, 24 * len(counts))),
                width="stretch",
            )

        st.markdown("#### ③ 변수 사이의 관계 (상관 분석)")
        if len(eda_num_cols) < 2:
            st.info("상관 분석에는 수치형 열이 2개 이상 필요합니다.")
        else:
            corr = clean_df[eda_num_cols].corr()
            corr_long = corr.rename_axis("변수1").reset_index().melt(id_vars="변수1", var_name="변수2", value_name="상관계수")
            base = alt.Chart(corr_long).encode(x=alt.X("변수1:N", title=None), y=alt.Y("변수2:N", title=None))
            heatmap = base.mark_rect().encode(
                color=alt.Color("상관계수:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True)),
            )
            labels = base.mark_text(fontSize=12).encode(
                text=alt.Text("상관계수:Q", format=".2f"),
                color=alt.condition("abs(datum['상관계수']) > 0.6", alt.value("white"), alt.value("black")),
            )
            st.altair_chart((heatmap + labels).properties(height=max(220, 50 * len(eda_num_cols))), width="stretch")

            pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
            if not pairs.empty:
                (best_a, best_b), best_r = max(pairs.items(), key=lambda item: abs(item[1]))
                strength = "강한" if abs(best_r) >= 0.7 else "보통의" if abs(best_r) >= 0.4 else "약한"
                direction = "양(+)의" if best_r > 0 else "음(-)의"
                st.write(f"- 가장 관련이 큰 변수 쌍은 **{best_a}**와 **{best_b}**이며, {strength} {direction} 상관관계(r = {best_r:.2f})를 보입니다.")

            s1, s2 = st.columns(2)
            sx = s1.selectbox("X 변수", eda_num_cols, key=wkey("eda_sx"))
            sy = s2.selectbox("Y 변수", eda_num_cols, index=1, key=wkey("eda_sy"))
            points_chart = alt.Chart(clean_df).mark_circle(size=60, opacity=0.6).encode(
                x=alt.X(fld(sx), title=str(sx), scale=alt.Scale(zero=False)),
                y=alt.Y(fld(sy), title=str(sy), scale=alt.Scale(zero=False)),
            )
            trend = points_chart.transform_regression(fld(sx), fld(sy)).mark_line(color="#b66b3d", size=3)
            st.altair_chart((points_chart + trend).properties(height=320), width="stretch")
            st.write(f"- {sx}와 {sy}의 상관계수: **{clean_df[sx].corr(clean_df[sy]):.2f}**")

        st.markdown("#### ④ 그룹별 비교")
        group_options = [c for c in eda_cat_cols if clean_df[c].nunique() <= 30]
        if not group_options or not eda_num_cols:
            st.info("그룹 비교에는 범주가 30개 이하인 범주형 열과 수치형 열이 필요합니다.")
        else:
            g1, g2 = st.columns(2)
            group_col = g1.selectbox("그룹 기준(범주형)", group_options, key=wkey("eda_group"))
            value_col = g2.selectbox("비교할 값(수치형)", eda_num_cols, key=wkey("eda_value"))
            group_stats = clean_df.groupby(group_col)[value_col].agg(["count", "mean", "median", "std"]).round(2)
            group_stats.columns = ["개수", "평균", "중앙값", "표준편차"]
            st.dataframe(group_stats, width="stretch")
            top_group = group_stats["평균"].idxmax()
            st.write(f"- {value_col}의 평균이 가장 높은 그룹은 **{top_group}** ({group_stats.loc[top_group, '평균']:,.2f})입니다.")

        st.markdown("#### ⑤ 분석 결과 정리")
        st.text_area(
            "EDA로 발견한 점과 탐구 질문에 대한 나의 결론을 적어 보세요",
            placeholder="예) 공부시간과 시험점수는 양의 상관관계가 있었다. 스마트폰 사용 시간은 ...",
            key=wkey("eda_conclusion"),
        )

with reg_tab:
    step_header("회귀 분석", "학습 시간과 성적의 관계를 통해 예측 모델을 이해합니다.")
    study_df = pd.DataFrame({
        "학습시간": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "시험점수": [42, 50, 58, 61, 67, 74, 81, 86, 92, 98],
    })
    slope, intercept = compute_regression(study_df)
    time_value = st.slider("학습시간 입력", min_value=1, max_value=12, value=6)
    pred_score = intercept + slope * time_value
    st.metric("회귀식", f"점수 = {intercept:.2f} + {slope:.2f} × 학습시간")
    st.metric("예측 점수", f"{pred_score:.1f}점")
    st.line_chart(study_df.set_index("학습시간")["시험점수"])

with cluster_tab:
    step_header("군집 분석", "거리를 기준으로 비슷한 데이터끼리 묶어 봅니다.")
    k_value = st.slider("군집 수 k", min_value=2, max_value=5, value=3)
    points = [
        (1.0, 2.0), (1.3, 2.1), (2.2, 3.0), (5.5, 6.0),
        (5.8, 5.9), (6.2, 6.3), (10.0, 9.5), (9.8, 10.2),
        (10.4, 9.8), (11.0, 11.0)
    ]
    cluster_labels = kmeans(points, k_value)
    cluster_df = pd.DataFrame({"x": [p[0] for p in points], "y": [p[1] for p in points], "군집": cluster_labels})
    st.scatter_chart(cluster_df, x="x", y="y")
    st.dataframe(cluster_df, width="stretch")
    st.write("k-평균 군집 분석은 거리를 기준으로 비슷한 데이터끼리 묶는 방법입니다.")

with assoc_tab:
    step_header("연관 분석", "함께 구매되는 상품의 규칙을 지지도와 신뢰도로 찾아봅니다.")
    basket_df = pd.DataFrame({
        "거래ID": [1, 2, 3, 4, 5, 6],
        "구매목록": [
            "우유, 빵, 시리얼",
            "우유, 바나나",
            "빵, 우유, 요구르트",
            "시리얼, 바나나",
            "우유, 시리얼, 빵",
            "빵, 요구르트",
        ],
    })
    item_a = st.selectbox("조건 아이템", ["우유", "빵", "시리얼", "바나나", "요구르트"])
    item_b = st.selectbox("결과 아이템", ["빵", "우유", "시리얼", "바나나", "요구르트"])

    transactions = [row["구매목록"].split(", ") for _, row in basket_df.iterrows()]
    total = len(transactions)
    a_count = sum(1 for items in transactions if item_a in items)
    ab_count = sum(1 for items in transactions if item_a in items and item_b in items)
    support = ab_count / total if total else 0
    confidence = ab_count / a_count if a_count else 0

    result_df = pd.DataFrame([
        {"조건": item_a, "결과": item_b, "지지도": round(support, 3), "신뢰도": round(confidence, 3)},
    ])
    st.table(result_df)
    st.write("- 지지도: 전체 거래 중 조건과 결과가 함께 발생하는 비율")
    st.write("- 신뢰도: 조건이 발생했을 때 결과가 발생할 확률")

with code_tab:
    step_header("💻 코드 실습", "주피터 노트북처럼 CSV 파일을 업로드하고, 파이썬 코드를 셀에 입력해 실행 결과를 바로 확인합니다.")
    nb = notebook_state()
    if raw_df is not None:
        nb["nb_files"]["collected_data.csv"] = raw_df.to_csv(index=False).encode("utf-8-sig")

    file_col, cell_col = st.columns([0.27, 0.73], gap="medium")
    with file_col:
        with st.container(border=True, key="nb_file_panel"):
            st.markdown("**📁 파일**")
            uploads = st.file_uploader("CSV 업로드", type=["csv", "txt"], accept_multiple_files=True,
                                       key="nb_uploader", label_visibility="collapsed")
            for upload in uploads or []:
                nb["nb_files"][upload.name] = upload.getvalue()
            if not nb["nb_files"]:
                st.caption("업로드한 파일이 여기에 표시됩니다.")
            for name, data in sorted(nb["nb_files"].items()):
                f1, f2 = st.columns([0.78, 0.22])
                f1.markdown(f"📄 `{name}`  \n<small>{len(data) / 1024:,.1f} KB</small>", unsafe_allow_html=True)
                f2.download_button("⬇", data, file_name=name, key=f"nb_dl_{name}", help="내려받기")
            if "collected_data.csv" in nb["nb_files"]:
                st.caption("`collected_data.csv`는 02단계에서 불러온 데이터입니다.")
            st.caption("코드에서는 `pd.read_csv('파일이름.csv')`로 불러옵니다.")

    with cell_col:
        first_file = next(iter(sorted(n for n in nb["nb_files"] if n.endswith(".csv"))), "")
        first_id = nb["nb_cells"][0]["id"]
        # 첫 셀이 비어 있거나 안내 문구 그대로면, 업로드한 파일을 불러오는 예시 코드로 채운다.
        if nb.get(f"nb_code_{first_id}", notebook_default_code("")) in ("", notebook_default_code("")):
            nb[f"nb_code_{first_id}"] = notebook_default_code(first_file)

        t1, t2, t3, _ = st.columns([0.2, 0.2, 0.2, 0.4])
        t1.button("＋ 코드 셀", on_click=notebook_add_cell, width="stretch", key="nb_add")
        t2.button("▶ 모두 실행", type="primary", width="stretch", key="nb_run_all",
                  on_click=notebook_run_upto, args=(len(nb["nb_cells"]) - 1,))
        t3.button("↺ 초기화", on_click=notebook_reset, width="stretch", key="nb_reset")

        for position, cell in enumerate(nb["nb_cells"]):
            with st.container(border=True, key=f"nb_cell_{cell['id']}"):
                c1, c2, c3 = st.columns([0.08, 0.84, 0.08])
                c1.button("▶", key=f"nb_run_{cell['id']}", help="이 셀까지 실행 (Ctrl+Enter로 입력 확정)",
                          on_click=notebook_run_upto, args=(position,), type="primary")
                c1.caption(f"[{cell['count'] or ' '}]")
                c2.text_area("코드", key=f"nb_code_{cell['id']}", height=130, label_visibility="collapsed",
                             placeholder="파이썬 코드를 입력하세요.")
                c3.button("🗑", key=f"nb_del_{cell['id']}", help="셀 삭제", on_click=notebook_delete_cell, args=(position,))
                if cell["output"]:
                    with st.container(key=f"nb_out_{cell['id']}"):
                        show_cell_output(cell["output"])

        st.caption(
            f"셀 사이에 변수를 이어 쓰기 위해, 셀을 실행하면 1번 셀부터 그 셀까지 차례로 다시 실행됩니다. "
            f"실행 시간은 {NOTEBOOK_TIMEOUT}초로 제한되며 pandas·numpy·matplotlib 등 데이터 분석 모듈만 import할 수 있습니다. "
            "마지막 줄의 값(예: data.describe())은 표로 표시됩니다."
        )

st.markdown("---")
st.subheader("📘 교과서 목차 정리")
for section_name, items in TEXTBOOK_CHAPTERS.items():
    with st.expander(section_name, expanded=True):
        for item in items:
            st.markdown(f"- {item}")

st.download_button(
    label="교과서 목차 다운로드",
    data=pd.DataFrame(
        [{"단원": section_name, "내용": item} for section_name, items in TEXTBOOK_CHAPTERS.items() for item in items]
    ).to_csv(index=False, encoding="utf-8-sig"),
    file_name="data_analysis_textbook_outline.csv",
    mime="text/csv",
)
