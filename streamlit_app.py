import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="자운고등학교 정보교사",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');
    :root {
        --ink: #182821;
        --forest: #234637;
        --moss: #dce8d9;
        --lime: #d6ee76;
        --paper: #f6f7f1;
        --muted: #68766e;
        --line: #dce2d9;
    }
    html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: #edf1e9; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] h1 { font-size: 1.12rem; }
    .block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 4rem; }
    .eyebrow { color: var(--forest); font: 500 .74rem 'DM Mono', monospace; letter-spacing: .08em; text-transform: uppercase; }
    .hero {
        background: var(--forest); color: #f6f7f1; padding: 2.7rem 3rem;
        border-radius: 8px; position: relative; overflow: hidden;
        animation: arrive .55s ease-out both;
    }
    .hero::after {
        content: ''; position: absolute; right: -52px; top: -88px; width: 250px; height: 250px;
        border: 1px solid rgba(214,238,118,.35); border-radius: 50%;
        box-shadow: 0 0 0 26px rgba(214,238,118,.06), 0 0 0 53px rgba(214,238,118,.05);
    }
    .hero .eyebrow { color: var(--lime); }
    .hero h1 { color: #fff; font-size: clamp(2rem, 4vw, 3.2rem); line-height: 1.28; margin: .8rem 0; max-width: 760px; }
    .hero p { color: #dbe6dc; max-width: 680px; font-size: 1.02rem; line-height: 1.8; margin-bottom: 0; }
    .section-label { color: var(--muted); font: 500 .72rem 'DM Mono', monospace; }
    .note-box { border-left: 3px solid #95b55a; padding: .25rem 0 .25rem 1rem; color: var(--muted); line-height: 1.8; }
    .stMetric { background: #fff; border: 1px solid var(--line); padding: 1rem 1.1rem; border-radius: 6px; }
    .stMetric label { color: var(--muted) !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 1.25rem; }
    .stTabs [data-baseweb="tab"] { height: 3rem; }
    div[data-testid="stForm"] { border-color: var(--line); background: #fff; }
    .course-kicker { color: var(--forest); font: 500 .72rem 'DM Mono', monospace; }
    @keyframes arrive { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    @media (max-width: 700px) {
        .hero { padding: 2rem 1.4rem; }
        .hero h1 { font-size: 2rem; }
        .block-container { padding-top: 1.2rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

COURSES = {
    "고등학교 정보": {
        "tag": "01 / DIGITAL LITERACY",
        "summary": "디지털 세상을 이해하고, 컴퓨팅 사고로 생활 속 문제를 해결합니다.",
        "learn": ["자료와 정보의 표현", "알고리즘과 프로그래밍", "컴퓨팅 시스템과 네트워크", "디지털 사회와 정보 윤리"],
        "project": "학교생활의 불편을 찾아 순서도와 간단한 프로그램으로 해결하기",
        "question": "컴퓨터는 문제를 어떤 순서로 해결할까요?",
    },
    "인공지능 기초": {
        "tag": "02 / AI FOUNDATIONS",
        "summary": "인공지능의 원리를 익히고, 가능성과 한계를 비판적으로 살펴봅니다.",
        "learn": ["인공지능의 이해", "데이터와 기계학습", "문제 해결과 탐색", "AI 윤리와 사회적 영향"],
        "project": "생활 속 분류 문제를 정하고, 데이터와 모델의 판단 과정 설명하기",
        "question": "인공지능의 예측을 우리는 언제 믿어도 될까요?",
    },
    "데이터과학": {
        "tag": "03 / DATA SCIENCE",
        "summary": "질문에 맞는 데이터를 모으고 분석해 근거 있는 이야기로 전달합니다.",
        "learn": ["데이터 수집과 정제", "표와 그래프로 표현하기", "패턴과 관계 해석", "개인정보와 데이터 윤리"],
        "project": "관심 있는 학교·지역 주제를 데이터로 탐구하고 시각화하기",
        "question": "데이터는 우리 주변의 어떤 이야기를 들려줄까요?",
    },
}

with st.sidebar:
    st.markdown("## 자운고 정보교실")
    st.caption("정보 · 인공지능 기초 · 데이터과학")
    st.divider()
    page = st.radio(
        "둘러보기",
        ["교사 소개", "교과 안내", "데이터 실습"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("### 가르치는 사람")
    st.markdown("**자운고등학교 정보교사**")
    st.markdown(
        '<div class="note-box">기술을 익히는 데서 그치지 않고, '
        '기술로 더 나은 질문을 만드는 수업을 지향합니다.</div>',
        unsafe_allow_html=True,
    )


def render_hero(kicker, title, description):
    st.markdown(
        f'<section class="hero"><div class="eyebrow">{kicker}</div>'
        f'<h1>{title}</h1><p>{description}</p></section>',
        unsafe_allow_html=True,
    )


if page == "교사 소개":
    render_hero(
        "JAWOON HIGH SCHOOL AI교실 HanLAB/ INFORMATICS",
        "질문에서 시작해,<br>데이터로 답을 찾습니다.",
        "안녕하세요. 자운고등학교에서 정보, 인공지능 기초, 데이터과학을 가르칩니다. "
        "직접 만들고 실험하며 디지털 세상을 주도적으로 읽는 힘을 함께 기릅니다.",
    )
    st.write("")
    st.markdown('<div class="section-label">CLASSROOM AT A GLANCE</div>', unsafe_allow_html=True)
    metric_cols = st.columns(3)
    metric_cols[0].metric("함께 배우는 교과", "03", "정보 · AI · 데이터")
    metric_cols[1].metric("수업의 출발점", "질문", "생활 속 문제에서")
    metric_cols[2].metric("배움의 방식", "만들기", "탐구하고 나누기")
    st.write("")
    st.subheader("기술을 배우고, 세상을 읽는 수업")
    value_cols = st.columns(3, gap="large")
    values = [
        ("01", "직접 해보기", "개념을 코드와 작은 실험으로 바꾸며 원리를 몸으로 익힙니다."),
        ("02", "근거로 말하기", "데이터를 살펴보고, 해석의 근거와 한계를 함께 설명합니다."),
        ("03", "책임 있게 쓰기", "개인정보와 편향을 살피며 기술이 미치는 영향을 생각합니다."),
    ]
    for column, (number, title, detail) in zip(value_cols, values):
        with column:
            st.markdown(f'<div class="course-kicker">{number} / OUR APPROACH</div>', unsafe_allow_html=True)
            st.markdown(f"#### {title}")
            st.write(detail)
    st.divider()
    st.subheader("세 교과, 하나의 탐구 흐름")
    tabs = st.tabs(list(COURSES))
    for tab, (name, course) in zip(tabs, COURSES.items()):
        with tab:
            st.markdown(f'<div class="course-kicker">{course["tag"]}</div>', unsafe_allow_html=True)
            st.markdown(f"### {name}")
            st.write(course["summary"])
            st.markdown(f'**탐구 질문**　{course["question"]}')
            st.caption(f'프로젝트 예시 · {course["project"]}')

elif page == "교과 안내":
    render_hero(
        "COURSE GUIDE / 2026",
        "배움의 지도",
        "각 교과에서 무엇을 배우고 어떤 질문을 탐구하는지 살펴보세요.",
    )
    st.write("")
    tabs = st.tabs(list(COURSES))
    for tab, (name, course) in zip(tabs, COURSES.items()):
        with tab:
            left, right = st.columns([1.15, 0.85], gap="large")
            with left:
                st.markdown(f'<div class="course-kicker">{course["tag"]}</div>', unsafe_allow_html=True)
                st.markdown(f"## {name}")
                st.write(course["summary"])
                st.markdown("#### 이런 질문을 탐구해요")
                st.info(course["question"])
                st.markdown("#### 프로젝트 예시")
                st.write(course["project"])
            with right:
                st.markdown("#### 핵심 배움")
                for index, topic in enumerate(course["learn"], start=1):
                    st.markdown(f"**{index:02d}**　{topic}")
                    if index < len(course["learn"]):
                        st.divider()
    outline = pd.DataFrame(
        [{"교과": name, "탐구 질문": course["question"], "프로젝트": course["project"]}
         for name, course in COURSES.items()]
    )
    st.download_button(
        "교과 안내표 다운로드",
        data=outline.to_csv(index=False).encode("utf-8-sig"),
        file_name="jaun_course_guide.csv",
        mime="text/csv",
        icon=":material/download:",
    )
    with st.expander("수업에서 중요하게 생각하는 것"):
        st.write(
            "정답을 빠르게 찾는 것만큼, 좋은 질문을 세우고 과정을 설명하는 일을 중요하게 생각합니다. "
            "협업과 성찰을 통해 배운 내용을 새로운 문제에 적용해 봅니다."
        )

else:
    render_hero(
        "DATA LAB / TRY IT YOURSELF",
        "숫자를 그래프로,<br>그래프를 질문으로.",
        "아래 점수를 직접 조정해 간단한 데이터 요약과 시각화를 만들어 보세요. "
        "입력한 값은 이 화면 안에서만 사용됩니다.",
    )
    st.write("")
    st.subheader("나만의 과목별 점수 살펴보기")
    st.caption("슬라이더 값을 바꾸면 요약과 그래프가 함께 업데이트됩니다.")
    score_cols = st.columns(3)
    scores = {}
    for column, subject, initial in zip(score_cols, ["정보", "인공지능 기초", "데이터과학"], [82, 74, 91]):
        with column:
            scores[subject] = st.slider(f"{subject} 점수", 0, 100, initial)
    score_df = pd.DataFrame({"과목": list(scores), "점수": list(scores.values())}).set_index("과목")
    average = sum(scores.values()) / len(scores)
    highest = max(scores, key=scores.get)
    summary_cols = st.columns(3)
    summary_cols[0].metric("입력한 과목", f"{len(scores)}개")
    summary_cols[1].metric("산술 평균", f"{average:.1f}점")
    summary_cols[2].metric("가장 높은 점수", highest, f"{scores[highest]}점")
    st.bar_chart(score_df, y="점수", color="#52785a", height=300)
    with st.expander("표 데이터 확인"):
        st.dataframe(score_df.reset_index(), hide_index=True, width="stretch")

    st.divider()
    st.subheader("한 문제로 확인하는 AI 기초")
    with st.form("ai_check_form"):
        answer = st.radio(
            "기계학습에서 모델이 패턴을 찾는 데 사용하는 것은 무엇일까요?",
            ["학습 데이터", "화면의 색상", "컴퓨터의 이름"],
            horizontal=True,
        )
        submitted = st.form_submit_button("답 확인", type="primary")
    if submitted:
        if answer == "학습 데이터":
            st.success("맞아요. 학습 데이터에서 패턴을 찾아 새로운 입력에 대한 예측을 만듭니다.")
        else:
            st.info("다시 생각해 볼까요? 모델은 학습 데이터에서 패턴을 찾습니다.")
    st.caption("점수는 실습을 위한 직접 입력값입니다. 평가나 성취 수준을 나타내지 않습니다.")
