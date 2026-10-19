from collections import Counter
import html
import io
import re

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


st.markdown('<div class="section-header">Ⅰ. 데이터 준비와 분석</div>', unsafe_allow_html=True)

st.markdown("### 01. 데이터 수집")
st.markdown('<div class="subbox">문제를 정의하고, 필요한 데이터를 수집하며, 수집된 데이터의 특성을 파악하는 과정입니다. '
            '여기서 불러온 데이터가 02. 전처리·시각화 → 03. 탐색적 데이터 분석으로 이어집니다.</div>', unsafe_allow_html=True)

research_question = st.text_input(
    "① 문제 정의: 이 데이터로 알아보고 싶은 질문을 적어 보세요",
    placeholder="예) 공부시간이 많을수록 시험점수가 높을까?",
)

source = st.radio("② 데이터 수집 방법", ["CSV 파일 업로드", "예시 데이터 사용"], horizontal=True)
raw_df = None
data_id = None
if source == "CSV 파일 업로드":
    collected_file = st.file_uploader("수집한 CSV 파일을 업로드하세요", type=["csv"], key="collect_csv")
    if collected_file is not None:
        try:
            raw_df = read_uploaded_csv(collected_file)
            data_id = f"{collected_file.name}_{collected_file.size}"
            st.success(f"{collected_file.name} 파일을 불러왔습니다.")
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            st.error(f"CSV 파일을 읽을 수 없습니다. {error}")
    else:
        st.info("CSV 파일을 업로드하면 아래 전처리·시각화·EDA 실습이 이 데이터로 진행됩니다.")
else:
    sample_name = st.selectbox("예시 데이터 선택", list(SAMPLE_DATASETS))
    raw_df = SAMPLE_DATASETS[sample_name]()
    data_id = sample_name


def wkey(name):
    # 데이터가 바뀌면 위젯 선택값도 새로 시작하도록 키에 데이터 식별자를 붙인다.
    return f"{name}_{data_id}"


if raw_df is not None:
    st.markdown("#### ③ 수집한 데이터의 특성")
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
            st.write(f"- 결측값이 있는 열: {', '.join(missing_cols)} → 전처리 필요")
        time_cols = [str(c) for c in raw_df.columns if re.search(r"연도|년|날짜|year|date", str(c), re.IGNORECASE)]
        if time_cols:
            st.write(f"- 시간 관련 열({', '.join(time_cols)})이 있어 시계열 분석이 가능합니다.")
        st.markdown("<div class='note'>데이터를 수집한 뒤에는 '이 데이터로 내 질문을 해결할 수 있는가?'를 판단하는 과정이 꼭 필요합니다.</div>", unsafe_allow_html=True)

st.markdown('<div class="section-header">02. 데이터 전처리와 시각화</div>', unsafe_allow_html=True)

clean_df = None
if raw_df is None:
    st.info("01. 데이터 수집에서 CSV 파일을 업로드하거나 예시 데이터를 선택하면 전처리 실습을 시작할 수 있습니다.")
else:
    st.markdown("### 데이터 전처리")
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

    st.markdown("### 데이터 시각화")
    num_cols, cat_cols = split_columns(clean_df)
    if clean_df.empty:
        st.warning("전처리 후 남은 데이터가 없습니다. 전처리 조건을 조정해 보세요.")
    elif not num_cols:
        st.warning("시각화하려면 수치형 열이 하나 이상 필요합니다. ② 숫자로 변환할 열을 확인해 보세요.")
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
                st.caption("범주가 많은 열은 ⑥ 행 필터로 몇 개 값만 남기면 색상 구분에 사용할 수 있습니다.")
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
    st.dataframe(frequency_df, hide_index=True, width="stretch")
elif analysis_text is not None:
    st.info("두 글자 이상의 단어를 입력하면 워드클라우드가 표시됩니다.")

st.markdown('<div class="section-header">03. 데이터 분석 방법</div>', unsafe_allow_html=True)

anal_tabs = st.tabs(["탐색적 데이터 분석", "회귀 분석", "군집 분석", "연관 분석"])

with anal_tabs[0]:
    st.markdown("### 탐색적 데이터 분석(EDA)")
    if clean_df is None or clean_df.empty:
        st.info("01. 데이터 수집과 02. 전처리를 마치면 전처리된 데이터로 EDA를 진행합니다.")
    else:
        eda_num_cols, eda_cat_cols = split_columns(clean_df)
        if research_question:
            st.markdown(f"<div class='note'>🔎 탐구 질문: {html.escape(research_question)}</div>", unsafe_allow_html=True)
        st.caption(f"02. 전처리를 거친 데이터 {len(clean_df):,}행 × {len(clean_df.columns)}열을 분석합니다.")

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
    st.dataframe(cluster_df, width="stretch")
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
