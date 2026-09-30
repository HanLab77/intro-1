from collections import Counter
import html
import io
import re

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
    st.button("데이터 새로고침", use_container_width=True)


@st.cache_data
def load_gdp_data():
    df = pd.read_csv("data/gdp_data.csv")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"Country Name": "국가", "Country Code": "국가코드", "Indicator Name": "지표명"})
    return df


def make_country_df(country_name, year_start, year_end):
    df = load_gdp_data()
    country_df = df[df["국가"] == country_name].copy()
    if country_df.empty:
        return pd.DataFrame(columns=["연도", "GDP(현재 US$)"])

    year_cols = [str(y) for y in range(year_start, year_end + 1)]
    available_years = [col for col in year_cols if col in country_df.columns]
    long_df = country_df[["국가", *available_years]].melt(
        id_vars=["국가"], value_vars=available_years, var_name="연도", value_name="GDP(현재 US$)"
    )
    long_df["연도"] = long_df["연도"].astype(int)
    long_df["GDP(현재 US$)"] = pd.to_numeric(long_df["GDP(현재 US$)"], errors="coerce")
    return long_df.dropna(subset=["GDP(현재 US$)"]).sort_values("연도").reset_index(drop=True)


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


def score_summary(df):
    return {
        "행 개수": len(df),
        "컬럼 수": len(df.columns),
        "결측값 수": int(df.isnull().sum().sum()),
    }


st.markdown('<div class="section-header">Ⅰ. 데이터 준비와 분석</div>', unsafe_allow_html=True)

with st.container():
    left, right = st.columns([1.3, 0.7])
    with left:
        st.markdown("### 01. 데이터 수집")
        st.markdown('<div class="subbox">문제를 정의하고, 필요한 데이터를 수집하며, 수집된 데이터의 특성을 파악하는 과정입니다.</div>', unsafe_allow_html=True)
        gdp_df = load_gdp_data()
        selected_country = st.selectbox("국가를 선택하세요", gdp_df["국가"].drop_duplicates().tolist(), index=100)
        year_start, year_end = st.slider("연도 범위", min_value=1960, max_value=2022, value=(2000, 2022))
        chart_df = make_country_df(selected_country, year_start, year_end)

        if not chart_df.empty:
            st.metric("선택 국가", selected_country)
            first_val = chart_df["GDP(현재 US$)"].iloc[0]
            last_val = chart_df["GDP(현재 US$)"].iloc[-1]
            growth = ((last_val - first_val) / first_val) * 100 if first_val else 0
            col1, col2, col3 = st.columns(3)
            col1.metric("최초 연도 GDP", f"{first_val:,.0f}")
            col2.metric("최종 연도 GDP", f"{last_val:,.0f}")
            col3.metric("증감률", f"{growth:.1f}%")
            st.line_chart(chart_df.set_index("연도")["GDP(현재 US$)"])
        else:
            st.warning("선택한 국가의 GDP 데이터를 찾을 수 없습니다.")

    with right:
        st.markdown("### 데이터 특성 확인")
        st.write("- 국가별 GDP는 시계열 데이터이므로 시간의 흐름을 살펴보는 것이 주요 분석 대상입니다.")
        st.write("- 숫자 값이 많아 결측치와 이상치 탐지가 중요합니다.")
        st.write("- 연도, 국가, GDP 값은 데이터 수집 이후 전처리를 통해 분석에 적합한 형태로 바꾸어야 합니다.")
        st.markdown("<div class='note'>데이터를 수집한 뒤에는 '어떤 질문을 해결할 수 있는가?'를 판단하는 과정이 꼭 필요합니다.</div>", unsafe_allow_html=True)

st.markdown('<div class="section-header">02. 데이터 전처리와 시각화</div>', unsafe_allow_html=True)

st.markdown("### 데이터 전처리")
uploaded = st.file_uploader("CSV 파일을 업로드해 보세요", type=["csv"], accept_multiple_files=False)
if uploaded is not None:
    raw = pd.read_csv(uploaded)
else:
    raw = load_gdp_data().head(20)

selected_cols = st.multiselect("기준이 되는 열을 선택하세요", raw.columns.tolist(), default=list(raw.columns[:5]))
cleaned = raw[selected_cols].copy()
cleaned = cleaned.dropna(subset=selected_cols)
st.dataframe(cleaned.head(10), use_container_width=True)
st.write(f"전처리 후 행 수: {len(cleaned)}")
st.bar_chart(cleaned.isnull().sum())

st.markdown("### 데이터 시각화")
vis_type = st.selectbox("그래프 유형", ["꺾은선형", "막대형", "산점도"])
chart_data = pd.DataFrame({
    "범주": ["수집", "정제", "정리", "시각화"],
    "값": [40, 75, 88, 96],
})
if vis_type == "꺾은선형":
    st.line_chart(chart_data.set_index("범주")["값"])
elif vis_type == "막대형":
    st.bar_chart(chart_data.set_index("범주")["값"])
else:
    scatter_df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6],
        "y": [5, 6, 8, 7, 9, 12],
    })
    st.scatter_chart(scatter_df, x="x", y="y")

st.markdown("### 워드클라우드 실습")
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
    st.dataframe(frequency_df, hide_index=True, use_container_width=True)
elif analysis_text is not None:
    st.info("두 글자 이상의 단어를 입력하면 워드클라우드가 표시됩니다.")

st.markdown('<div class="section-header">03. 데이터 분석 방법</div>', unsafe_allow_html=True)

anal_tabs = st.tabs(["탐색적 데이터 분석", "회귀 분석", "군집 분석", "연관 분석"])

with anal_tabs[0]:
    st.markdown("### 탐색적 데이터 분석(EDA)")
    eda_country = st.selectbox("기준 국가 선택", load_gdp_data()["국가"].drop_duplicates().tolist(), index=30)
    eda_df = make_country_df(eda_country, 2000, 2022)
    if not eda_df.empty:
        st.dataframe(eda_df.head(10), use_container_width=True)
        summary = eda_df["GDP(현재 US$)"].describe().to_frame(name="값")
        st.table(summary)
        st.line_chart(eda_df.set_index("연도")["GDP(현재 US$)"])

with anal_tabs[1]:
    st.markdown("### 회귀 분석")
    st.write("학습 시간과 성적의 관계를 통해 예측 모델을 이해합니다.")
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

with anal_tabs[2]:
    st.markdown("### 군집 분석")
    k_value = st.slider("군집 수 k", min_value=2, max_value=5, value=3)
    points = [
        (1.0, 2.0), (1.3, 2.1), (2.2, 3.0), (5.5, 6.0),
        (5.8, 5.9), (6.2, 6.3), (10.0, 9.5), (9.8, 10.2),
        (10.4, 9.8), (11.0, 11.0)
    ]
    cluster_labels = kmeans(points, k_value)
    cluster_df = pd.DataFrame({"x": [p[0] for p in points], "y": [p[1] for p in points], "군집": cluster_labels})
    st.scatter_chart(cluster_df, x="x", y="y")
    st.dataframe(cluster_df, use_container_width=True)
    st.write("k-평균 군집 분석은 거리를 기준으로 비슷한 데이터끼리 묶는 방법입니다.")

with anal_tabs[3]:
    st.markdown("### 연관 분석")
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
