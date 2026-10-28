import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_diabetes, load_iris, make_blobs, make_moons
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, pairwise_distances, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz


st.set_page_config(page_title="인공지능 기초 실험실", page_icon="🧠", layout="wide")

st.markdown(
	"""
	<style>
	@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');
	:root { --ink:#182821; --forest:#234637; --lime:#d6ee76; --paper:#f6f7f1; --muted:#68766e; --line:#dce2d9; }
	html, body, [class*="css"] { font-family:'Noto Sans KR', sans-serif; }
	.stApp { background:var(--paper); color:var(--ink); }
	.block-container { max-width:1180px; padding-top:2rem; padding-bottom:4rem; }
	.lab-header { background:var(--forest); color:var(--paper); padding:2rem 2.4rem; border-radius:8px; animation:arrive .5s ease-out both; }
	.lab-kicker { color:var(--lime); font:500 .74rem 'DM Mono',monospace; }
	.lab-header h1 { color:#fff; font-size:2rem; margin:.5rem 0; }
	.lab-header p { color:#dbe6dc; margin:0; line-height:1.8; max-width:780px; }
	.section-label { color:var(--forest); font:500 .74rem 'DM Mono',monospace; }
	.coach-note { border-left:3px solid #95b55a; padding:.2rem 0 .2rem 1rem; color:var(--muted); line-height:1.8; }
	div[data-testid="stForm"] { background:#fff; border-color:var(--line); }
	.stMetric { background:#fff; border:1px solid var(--line); padding:1rem; border-radius:6px; }
	@keyframes arrive { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
	@media (max-width:700px) { .lab-header { padding:1.5rem; } .lab-header h1 { font-size:1.6rem; } .block-container { padding-top:1rem; } }
	</style>
	""",
	unsafe_allow_html=True,
)

st.markdown(
	'<header class="lab-header"><div class="lab-kicker">JAUN HIGH SCHOOL / AI FOUNDATIONS</div>'
	'<h1>기계학습 실험실</h1>'
	'<p>문제에 맞는 학습 방법을 고르고, 데이터를 나누어 모델을 훈련한 뒤, '
	'예측 결과와 성능을 직접 확인해 봅니다.</p></header>',
	unsafe_allow_html=True,
)

# ── 01 · 기계학습의 유형: 분류 알고리즘 체험용 데이터와 도우미 ─────────────────
AXIS_DOMAIN = [0, 10]
KNN_SHAPES = {"● 동그라미": ("circle", "#e0457b"), "▲ 세모": ("triangle-up", "#2f6fb5"), "■ 네모": ("square", "#4c9a3f")}
KNN_POINTS = pd.DataFrame(
	[(1.6, 8.6, "● 동그라미"), (2.7, 8.4, "● 동그라미"), (1.9, 7.3, "● 동그라미"), (0.8, 5.4, "● 동그라미"), (3.6, 7.1, "● 동그라미"),
	 (6.4, 7.8, "▲ 세모"), (7.4, 8.2, "▲ 세모"), (5.6, 6.5, "▲ 세모"), (6.9, 6.4, "▲ 세모"), (5.9, 5.4, "▲ 세모"), (7.1, 5.8, "▲ 세모"),
	 (1.2, 3.6, "■ 네모"), (2.2, 3.2, "■ 네모"), (3.1, 3.9, "■ 네모"), (2.5, 2.1, "■ 네모"), (4.4, 2.6, "■ 네모")],
	columns=["특성1", "특성2", "기호"],
)
CLASSIFY_QUIZ = [
	("이 메일이 스팸일 확률이 몇 %인지 알고 싶어요.", "로지스틱 회귀"),
	("고양이와 강아지 사진 사이에서 가장 안전한(넓은) 경계를 찾아 구분하고 싶어요.", "서포트 벡터 머신"),
	("나와 취향이 비슷한 친구 5명이 재미있게 본 영화를 추천받고 싶어요.", "k-최근접 이웃"),
	("연봉 → 신용등급 → 기존 대출 순서로 질문하며 대출 승인 여부를 정하고, 이유도 설명하고 싶어요.", "의사결정트리"),
	("공부 시간만 보고 합격(1)인지 불합격(0)인지 판정하고 싶어요.", "로지스틱 회귀"),
]


def sigmoid(z):
	return 1 / (1 + np.exp(-z))


def line_frame(nx, ny, c):
	# nx·x + ny·y = c 위의 두 점. 차트 범위 밖으로 나가는 부분은 clip으로 잘린다.
	if abs(ny) > 1e-9:
		xs = np.array([-20.0, 30.0])
		return pd.DataFrame({"x": xs, "y": (c - nx * xs) / ny})
	return pd.DataFrame({"x": [c / nx] * 2, "y": [-20.0, 30.0]})


def axis(name):
	return alt.Scale(domain=AXIS_DOMAIN, nice=False), alt.Axis(title=name)


@st.cache_data
def svm_animals():
	rng = np.random.default_rng(11)
	deer = rng.normal([3.0, 3.8], [0.75, 1.0], (10, 2))
	tiger = rng.normal([7.0, 6.6], [0.75, 0.9], (9, 2))
	frame = pd.DataFrame(np.vstack([deer, tiger]).clip(0.6, 9.4), columns=["크기", "속도"])
	frame["동물"] = ["사슴"] * len(deer) + ["호랑이"] * len(tiger)
	frame["그림"] = frame["동물"].map({"사슴": "🦌", "호랑이": "🐯"})
	return frame


@st.cache_data
def admission_data():
	rng = np.random.default_rng(5)
	score = rng.uniform(55, 100, 80).round()
	attend = rng.uniform(70, 100, 80).round()
	passed = (score >= 80) & (attend >= 90)
	flip = rng.random(80) < 0.05
	return pd.DataFrame({"모의고사 점수": score, "출석률": attend, "결과": np.where(passed ^ flip, "합격", "불합격")})


@st.cache_data
def customer_blobs():
	points, _ = make_blobs(n_samples=150, centers=4, cluster_std=0.9, center_box=(1.5, 8.5), random_state=3)
	return pd.DataFrame(points.clip(0.2, 9.8), columns=["방문 빈도", "구매 금액"])


def animal_tree_dot(path):
	nodes = {
		"q1": "날개가 있나요?", "q2": "날 수 있나요?", "q3": "지느러미가\n있나요?",
		"매": "🦅 매", "펭귄": "🐧 펭귄", "돌고래": "🐬 돌고래", "곰": "🐻 곰",
	}
	edges = [("q1", "q2", "예"), ("q1", "q3", "아니요"), ("q2", "매", "예"), ("q2", "펭귄", "아니요"), ("q3", "돌고래", "예"), ("q3", "곰", "아니요")]
	lines = ['digraph { bgcolor="transparent"; node [shape=box style="rounded,filled" fontname="Noto Sans KR" fontsize=13];']
	for key, label in nodes.items():
		active = key in path
		fill = ("#1d4d3b" if key.startswith("q") else "#f2a541") if active else "#ffffff"
		font = "#ffffff" if active and key.startswith("q") else "#182821"
		lines.append(f'"{key}" [label="{label}" fillcolor="{fill}" fontcolor="{font}" color="#9fb59a"];')
	for a, b, label in edges:
		active = a in path and b in path
		lines.append(f'"{a}" -> "{b}" [label="{label}" color="{"#e0457b" if active else "#b9c4b5"}" penwidth={3 if active else 1}];')
	return "\n".join(lines) + "}"


CLUSTER_COLORS = ["#e04848", "#2f6fb5", "#4c9a3f", "#f2a541", "#7b4bb7", "#2fb4c4"]
FOOD_ITEMS = [("🍌", "바나나", "과일", "길쭉한 모양"), ("🍆", "가지", "채소", "길쭉한 모양"), ("🍎", "사과", "과일", "동그란 모양"),
			  ("🥒", "오이", "채소", "길쭉한 모양"), ("🟠", "감", "과일", "동그란 모양")]
CLUSTER_OR_CLASSIFY = [
	("스팸/정상 정답이 붙은 메일로 학습해, 새 메일이 스팸인지 판별한다.", "분류"),
	("구매 기록만 가지고 비슷한 소비 습관의 고객 그룹을 찾는다.", "군집화"),
	("정답 없이 키와 성적이 비슷한 친구끼리 모둠을 만든다.", "군집화"),
	("'고양이', '강아지' 이름표가 붙은 사진으로 학습해 새 사진을 판별한다.", "분류"),
	("주제가 정해지지 않은 뉴스 기사들을 내용이 비슷한 것끼리 묶는다.", "군집화"),
]
KMEANS_QUIZ = [
	("비지도학습에 사용하는 데이터에는…", ["정답(레이블)이 있다", "정답(레이블)이 없다"], "정답(레이블)이 없다"),
	("k-평균 알고리즘에서 K는?", ["중심점의 좌표", "군집(모둠)의 개수", "반복 횟수"], "군집(모둠)의 개수"),
	("3단계에서 중심점은 어디로 이동할까?", ["가장 먼 데이터 위치", "군집에 속한 데이터들의 평균 위치", "그래프의 원점"], "군집에 속한 데이터들의 평균 위치"),
	("군집화는 언제 끝날까?", ["중심점이 더 이상 이동하지 않을 때", "정확히 10번 반복했을 때", "데이터가 모두 같은 군집이 될 때"], "중심점이 더 이상 이동하지 않을 때"),
	("k-평균의 단점은?", ["처음 중심점에 따라 결과가 달라질 수 있다", "계산이 매우 복잡하고 느리다"], "처음 중심점에 따라 결과가 달라질 수 있다"),
]
KMEANS_STAGES = [
	("1단계", "대표(중심점) 뽑기", "K개의 중심점을 임의로 정해요."),
	("2단계", "가까운 팀으로", "각 데이터는 가장 가까운 중심점의 팀에 소속돼요."),
	("3단계", "중심점 다시 뽑기", "팀 데이터들의 평균 위치로 중심점을 옮겨요."),
	("4단계", "반복", "중심점이 더 이상 움직이지 않으면 완료!"),
]


@st.cache_data
def classmates():
	rng = np.random.default_rng(21)
	groups = [((158, 62), (4, 6)), ((172, 88), (4, 5)), ((178, 60), (3.5, 6)), ((163, 90), (3.5, 4))]
	rows = np.vstack([rng.normal(center, spread, (9, 2)) for center, spread in groups])
	return pd.DataFrame({"키": rows[:, 0].round(1), "성적": rows[:, 1].clip(40, 100).round()})


@st.cache_data
def four_blobs():
	points, _ = make_blobs(n_samples=200, centers=[(2, 2.2), (2.6, 7.6), (7.6, 8), (7.8, 2.6)], cluster_std=0.75, random_state=2)
	return pd.DataFrame(points.clip(0.2, 9.8), columns=["특성1", "특성2"])


def kmeans_reset(k, seed, centers=None):
	data = customer_blobs().to_numpy()
	if centers is None:
		centers = data[np.random.default_rng(seed).choice(len(data), k, replace=False)].tolist()
	st.session_state["km"] = {"k": k, "seed": seed, "centers": [[float(a), float(b)] for a, b in centers], "prev": None,
							  "labels": None, "phase": "assign", "step": 0, "done": False}


def kmeans_step():
	km = st.session_state["km"]
	data = customer_blobs().to_numpy()
	centers = np.array(km["centers"])
	if km["phase"] == "assign":
		km["labels"] = np.argmin(((data[:, None, :] - centers[None]) ** 2).sum(-1), axis=1).tolist()
		km["phase"], km["prev"] = "move", None
	else:
		labels = np.array(km["labels"])
		moved = np.array([data[labels == i].mean(0) if (labels == i).any() else centers[i] for i in range(km["k"])])
		km["done"] = bool(np.allclose(moved, centers))
		km["prev"], km["centers"] = centers.tolist(), moved.tolist()
		km["phase"] = "assign"
		km["step"] += 1


def kmeans_finish():
	for _ in range(100):
		kmeans_step()
		if st.session_state["km"]["done"]:
			break


def kmeans_use_custom():
	custom = st.session_state.get("km_custom", [])
	kmeans_reset(len(custom), st.session_state["km"]["seed"], custom)
	st.session_state["km_custom"] = []

st.write("")
ml_types_tab, *tabs = st.tabs(["01 · 기계학습의 유형", "02 · 학습 유형과 알고리즘", "03 · 모델 만들기와 평가", "04 · 퍼셉트론과 딥러닝"])

with ml_types_tab:
	st.markdown('<div class="section-label">TYPES OF MACHINE LEARNING</div>', unsafe_allow_html=True)
	st.subheader("정답이 있으면 지도학습, 정답 없이 패턴을 찾으면 비지도학습")
	intro_left, intro_right = st.columns(2, gap="medium")
	with intro_left.container(border=True):
		st.markdown("#### 🎯 지도학습")
		st.write("입력과 **정답(레이블)** 이 함께 있는 데이터로 학습해, 새 데이터의 정답을 예측합니다.")
		st.caption("분류: 스팸/정상, 합격/불합격 · 회귀: 집값, 키")
	with intro_right.container(border=True):
		st.markdown("#### 🧩 비지도학습")
		st.write("**정답 없이** 입력 데이터만 보고, 비슷한 것끼리 묶거나 숨은 구조를 찾습니다.")
		st.caption("군집: 비슷한 소비 습관의 고객 묶기")

	sup_tab, unsup_tab = st.tabs(["🎯 지도학습", "🧩 비지도학습"])

	with sup_tab:
		intro_t, logi_t, svm_t, knn_t, tree_t, quiz_t = st.tabs(
			["① 분류란?", "② 로지스틱 회귀", "③ 서포트 벡터 머신", "④ k-최근접 이웃", "⑤ 의사결정트리", "⑥ 정리 퀴즈"]
		)

		with intro_t:
			st.markdown("#### 분류란? 데이터가 어떤 클래스(범주)에 속할지 예측하는 문제")
			st.markdown('<div class="coach-note">모델이 경계(결정 경계)를 학습하여, 새로운 데이터가 어떤 클래스에 속하는지 예측합니다. '
						'예를 들어 스팸 메일 분류 모델은 입력값을 바탕으로 이메일이 스팸인지 아닌지를 판단해 나눕니다.</div>',
						unsafe_allow_html=True)
			kinds = st.columns(3, gap="small")
			for col, (title, desc, example) in zip(kinds, [
				("이진 분류", "두 개의 클래스 중 하나를 예측", "✉️ 스팸 / 정상"),
				("다중 분류", "세 개 이상의 클래스 중 하나를 예측", "🍎 사과 / 🍌 바나나 / 🍇 포도"),
				("멀티라벨 분류", "하나의 데이터가 여러 클래스에 동시에 속함", "🐶 강아지 + 🌱 잔디 + ⚽ 공"),
			]):
				with col.container(border=True):
					st.markdown(f"**{title}**")
					st.write(desc)
					st.caption(example)

			st.markdown("#### 분류 모델이 하는 일: 확률을 계산하고, 가장 높은 클래스를 고른다")
			st.caption("메일 속 의심 단어 수를 바꾸어 보세요. 모델이 각 클래스의 확률을 출력하고, 확률이 큰 쪽으로 예측합니다.")
			words = st.slider("✉️ 메일 속 의심 단어(무료, 당첨, 클릭…) 개수", 0, 10, 6, key="spam_words")
			p_spam = float(sigmoid(1.1 * (words - 4)))
			bar_col, verdict_col = st.columns([0.72, 0.28], gap="medium")
			with bar_col:
				probs = pd.DataFrame({"클래스": ["P(스팸)", "P(정상)"], "확률": [p_spam, 1 - p_spam]})
				bars = alt.Chart(probs).encode(
					y=alt.Y("클래스:N", sort=None, title=None, axis=alt.Axis(labelFontSize=15, labelFontWeight="bold")),
					x=alt.X("확률:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%", title="확률 (두 확률의 합 = 1)")),
				)
				st.altair_chart(alt.layer(
					bars.mark_bar(cornerRadiusEnd=6, size=34).encode(
						color=alt.Color("클래스:N", scale=alt.Scale(domain=["P(스팸)", "P(정상)"], range=["#e04848", "#3b82d6"]), legend=None)),
					bars.mark_text(align="left", dx=6, fontSize=15, fontWeight="bold").encode(text=alt.Text("확률:Q", format=".2f")),
					alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[4, 4], color="#888").encode(x="x:Q"),
				).properties(height=150), width="stretch")
			verdict_col.metric("최종 예측", "🚫 스팸" if p_spam >= 0.5 else "✅ 정상")
			verdict_col.caption("가장 높은 확률을 갖는 클래스를 최종 예측으로 골라요. (점선 = 0.5)")

		with logi_t:
			st.markdown("#### 로지스틱 회귀: 공부 시간으로 합격 확률 예측하기")
			st.markdown('<div class="coach-note">선형식 z = wx + b를 시그모이드 함수에 넣어 0~1 사이의 확률로 바꾸고, '
						'0.5 이상이면 합격(1), 0.5보다 작으면 불합격(0)으로 분류합니다.</div>', unsafe_allow_html=True)
			data_col, chart_col = st.columns([0.35, 0.65], gap="medium")
			with data_col:
				st.caption("학습 데이터 (표를 고치거나 행을 추가해 보세요)")
				logi_data = st.data_editor(
					pd.DataFrame({"공부시간": [1.0, 2.0, 3.0, 5.0, 6.0, 8.0], "결과": [0, 0, 0, 1, 1, 1]}),
					num_rows="dynamic", hide_index=True, key="logi_data",
					column_config={
						"공부시간": st.column_config.NumberColumn(min_value=0.0, max_value=10.0, step=0.5),
						"결과": st.column_config.SelectboxColumn(options=[0, 1], help="0 = 불합격, 1 = 합격"),
					},
				).dropna()
				study_time = st.slider("📚 새 학생의 공부 시간", 0.0, 10.0, 4.0, 0.5, key="logi_time")
				compare_linear = st.checkbox("선형 회귀 직선과 비교하기", key="logi_linear")
			if logi_data["결과"].nunique() < 2:
				chart_col.warning("합격(1)과 불합격(0) 데이터가 모두 있어야 학습할 수 있어요.")
			else:
				x_train = logi_data[["공부시간"]].to_numpy(float)
				y_train = logi_data["결과"].astype(int).to_numpy()
				logi = LogisticRegression().fit(x_train, y_train)
				w, b = float(logi.coef_[0][0]), float(logi.intercept_[0])
				p_pass = float(logi.predict_proba([[study_time]])[0][1])
				boundary = -b / w if w else None
				with chart_col:
					curve = pd.DataFrame({"공부시간": np.linspace(0, 10, 101)})
					curve["합격 확률"] = logi.predict_proba(curve[["공부시간"]].to_numpy())[:, 1]
					layers = [
						alt.Chart(curve).mark_line(color="#7b4bb7", strokeWidth=3).encode(
							x=alt.X("공부시간:Q", scale=alt.Scale(domain=[0, 10])), y=alt.Y("합격 확률:Q", scale=alt.Scale(domain=[-0.15, 1.15]))),
						alt.Chart(logi_data).mark_circle(size=140, color="#2fb4c4").encode(x="공부시간:Q", y=alt.Y("결과:Q")),
						alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4], color="#999").encode(y="y:Q"),
						alt.Chart(pd.DataFrame({"공부시간": [study_time], "합격 확률": [p_pass]})).mark_point(
							size=320, filled=True, color="#e0457b", shape="diamond").encode(x="공부시간:Q", y="합격 확률:Q"),
					]
					if boundary is not None and 0 <= boundary <= 10:
						layers.append(alt.Chart(pd.DataFrame({"x": [boundary]})).mark_rule(color="#e0457b", strokeDash=[6, 3]).encode(x="x:Q"))
					if compare_linear:
						lin = LinearRegression().fit(x_train, y_train)
						curve["선형 회귀"] = lin.predict(curve[["공부시간"]].to_numpy())
						layers.append(alt.Chart(curve).mark_line(color="#f2a541", strokeDash=[2, 2]).encode(x="공부시간:Q", y="선형 회귀:Q"))
					st.altair_chart(alt.layer(*layers).properties(height=320), width="stretch")
					m1, m2, m3 = st.columns(3)
					m1.metric("합격 확률", f"{p_pass:.0%}")
					m2.metric("불합격 확률", f"{1 - p_pass:.0%}")
					m3.metric("예측 결과", "🎉 합격(1)" if p_pass >= 0.5 else "📘 불합격(0)")
					z = w * study_time + b
					st.latex(rf"z = {w:.2f}\times{study_time:g} {'+' if b >= 0 else '-'} {abs(b):.2f} = {z:.2f},\quad \sigma(z)=\frac{{1}}{{1+e^{{-z}}}} = {sigmoid(z):.2f}")
					if boundary is not None:
						st.caption(f"결정 경계(빨간 점선): 공부 시간 약 {boundary:.1f}시간 — 이보다 많이 공부하면 합격으로 예측합니다.")
					if compare_linear:
						st.caption("주황 점선(선형 회귀)은 0과 1의 범위를 벗어날 수 있어요. 그래서 분류에는 S자 곡선의 로지스틱 회귀를 씁니다.")
			with st.expander("🔍 시그모이드 함수 탐험하기"):
				z_value = st.slider("z 값", -6.0, 6.0, 0.0, 0.5, key="logi_z")
				zs = pd.DataFrame({"z": np.linspace(-6, 6, 121)})
				zs["σ(z)"] = sigmoid(zs["z"])
				st.altair_chart(
					(alt.Chart(zs).mark_line(color="#e04040").encode(x="z:Q", y="σ(z):Q")
					 + alt.Chart(pd.DataFrame({"z": [z_value], "σ(z)": [sigmoid(z_value)]})).mark_point(size=200, filled=True).encode(x="z:Q", y="σ(z):Q")
					 ).properties(height=220), width="stretch")
				st.write(f"σ({z_value:g}) = **{sigmoid(z_value):.3f}** — z가 아무리 크거나 작아도 결과는 항상 0과 1 사이예요.")

		with svm_t:
			st.markdown("#### 서포트 벡터 머신: 사슴과 호랑이를 가장 안전하게 나누는 직선 찾기")
			st.markdown('<div class="coach-note">슬라이더로 직접 직선을 그어 보세요. 모든 동물을 바르게 나누면서, '
						'경계와 가장 가까운 동물까지의 거리(<b>마진</b>)가 가장 넓은 직선이 최고의 경계입니다.</div>', unsafe_allow_html=True)
			animals = svm_animals()
			svm = SVC(kernel="linear", C=1000).fit(animals[["크기", "속도"]], animals["동물"])
			(sw1, sw2), sb = svm.coef_[0], float(svm.intercept_[0])
			svm_margin = 1 / float(np.hypot(sw1, sw2))

			ctrl, view = st.columns([0.32, 0.68], gap="medium")
			with ctrl:
				st.markdown("**✏️ 내 직선**")
				angle = st.slider("기울기(각도)", 0, 179, 120, key="svm_angle", help="직선이 가로축과 이루는 각도")
				shift = st.slider("위치 이동", -5.0, 5.0, 0.0, 0.1, key="svm_shift")
				st.markdown("**❓ 새로운 동물**")
				new_size = st.slider("크기", 0.5, 9.5, 5.0, 0.1, key="svm_new_size")
				new_speed = st.slider("속도", 0.5, 9.5, 4.5, 0.1, key="svm_new_speed")
				show_svm = st.toggle("SVM 정답 경계 보기", key="svm_show")

			theta = np.radians(angle)
			nx, ny = -np.sin(theta), np.cos(theta)
			c = nx * 5 + ny * 5 + shift
			side = np.sign(animals["크기"] * nx + animals["속도"] * ny - c)
			tiger_side = 1 if (side[animals["동물"] == "호랑이"] > 0).sum() >= (side[animals["동물"] == "호랑이"] < 0).sum() else -1
			mine = np.where(side == tiger_side, "호랑이", "사슴")
			errors = int((mine != animals["동물"]).sum())
			my_margin = float(np.abs(animals["크기"] * nx + animals["속도"] * ny - c).min()) if errors == 0 else 0.0
			new_mine = "호랑이" if np.sign(new_size * nx + new_speed * ny - c) == tiger_side else "사슴"
			new_svm = svm.predict(pd.DataFrame({"크기": [new_size], "속도": [new_speed]}))[0]

			with view:
				xs, xa = axis("크기")
				ys, ya = axis("속도")
				layers = [
					alt.Chart(line_frame(nx, ny, c)).mark_line(color="#f2a541", strokeWidth=3, clip=True).encode(x=alt.X("x:Q", scale=xs, axis=xa), y=alt.Y("y:Q", scale=ys, axis=ya)),
					alt.Chart(animals).mark_text(fontSize=24).encode(x=alt.X("크기:Q", scale=xs), y=alt.Y("속도:Q", scale=ys), text="그림:N", tooltip=["동물", "크기", "속도"]),
					alt.Chart(pd.DataFrame({"크기": [new_size], "속도": [new_speed], "t": ["❓"]})).mark_text(fontSize=28).encode(x=alt.X("크기:Q", scale=xs), y=alt.Y("속도:Q", scale=ys), text="t:N"),
				]
				if show_svm:
					for offset, dash in [(0, []), (1, [6, 4]), (-1, [6, 4])]:
						layers.append(alt.Chart(line_frame(sw1, sw2, offset - sb)).mark_line(color="#2f6fb5", strokeDash=dash, clip=True).encode(x=alt.X("x:Q", scale=xs), y=alt.Y("y:Q", scale=ys)))
					sv = pd.DataFrame(svm.support_vectors_, columns=["크기", "속도"])
					layers.append(alt.Chart(sv).mark_point(size=1100, color="#e0457b", strokeWidth=2, strokeDash=[3, 2]).encode(x=alt.X("크기:Q", scale=xs), y=alt.Y("속도:Q", scale=ys)))
				st.altair_chart(alt.layer(*layers).properties(height=400), width="stretch")
				legend = "주황 실선 = 내 직선"
				if show_svm:
					legend += " · 파란 실선 = SVM 결정 경계 · 파란 점선 = 마진 · 빨간 원 = 서포트 벡터"
				st.caption(legend)

			r1, r2, r3, r4 = st.columns(4)
			r1.metric("잘못 나뉜 동물", f"{errors}마리")
			r2.metric("내 직선의 마진", f"{my_margin:.2f}")
			r3.metric("SVM의 마진", f"{svm_margin:.2f}" if show_svm else "?")
			r4.metric("새로운 동물", f"{'🐯' if new_mine == '호랑이' else '🦌'} {new_mine}", help=f"SVM의 판단: {new_svm}")
			if errors:
				st.warning("아직 잘못 나뉜 동물이 있어요. 각도와 위치를 바꿔 모든 동물을 바르게 나눠 보세요.")
			elif my_margin >= svm_margin * 0.95:
				st.success("🎉 SVM이 찾은 최적의 경계와 거의 같은 직선이에요! 마진이 최대가 되었습니다.")
				st.balloons()
			else:
				st.info(f"모두 바르게 나눴어요! 이제 마진을 더 넓혀 보세요. (목표의 {my_margin / svm_margin:.0%} 달성)")
			if new_mine != new_svm:
				st.caption(f"❓ 새로운 동물을 내 직선은 **{new_mine}**, SVM은 **{new_svm}**(으)로 분류했어요. 마진이 좁으면 새 데이터를 잘못 분류하기 쉬워요.")
			with st.expander("📖 용어 정리"):
				st.markdown("- **결정 경계(Decision Boundary)**: 데이터를 나누는 선(또는 면)\n"
							"- **마진(Margin)**: 경계선과 가장 가까운 데이터까지의 거리\n"
							"- **서포트 벡터(Support Vector)**: 경계선(마진)을 결정하는 가장 가까운 데이터")

		with knn_t:
			st.markdown("#### k-최근접 이웃: 가장 가까운 k개의 이웃에게 물어보기")
			st.markdown('<div class="coach-note">그래프를 <b>클릭</b>하면 그 자리에 새 데이터(✚)가 놓입니다. '
						'k를 바꾸면서 다수결 결과가 어떻게 달라지는지 확인해 보세요.</div>', unsafe_allow_html=True)
			st.session_state.setdefault("knn_x", 4.3)
			st.session_state.setdefault("knn_y", 6.5)
			k = st.slider("k (이웃의 수)", 1, 15, 3, key="knn_k")
			point = np.array([st.session_state["knn_x"], st.session_state["knn_y"]])
			distances = np.hypot(KNN_POINTS["특성1"] - point[0], KNN_POINTS["특성2"] - point[1])
			order = np.argsort(distances.to_numpy(), kind="stable")[:k]
			neighbors = KNN_POINTS.iloc[order]
			radius = float(distances.iloc[order[-1]])
			counts = neighbors["기호"].value_counts().reindex(list(KNN_SHAPES), fill_value=0)
			top = counts[counts == counts.max()].index.tolist()
			winner = top[0] if len(top) == 1 else next(lbl for lbl in neighbors["기호"] if lbl in top)

			xs, xa = axis("특성 1")
			ys, ya = axis("특성 2")
			grid = pd.DataFrame([(gx, gy) for gx in np.arange(0, 10.01, 0.25) for gy in np.arange(0, 10.01, 0.25)], columns=["특성1", "특성2"])
			pick = alt.selection_point(name="pick", fields=["특성1", "특성2"], on="click", nearest=True)
			circle = pd.DataFrame({"t": np.linspace(0, 2 * np.pi, 120)})
			circle["특성1"] = point[0] + radius * np.cos(circle["t"])
			circle["특성2"] = point[1] + radius * np.sin(circle["t"])
			enc = dict(x=alt.X("특성1:Q", scale=xs, axis=xa), y=alt.Y("특성2:Q", scale=ys, axis=ya))
			chart = alt.layer(
				alt.Chart(grid).mark_point(opacity=0.001, size=60).encode(**enc).add_params(pick),
				alt.Chart(circle).mark_line(color="#e0457b", strokeDash=[6, 4], clip=True).encode(**enc, order="t:Q"),
				alt.Chart(neighbors).mark_point(size=650, color="#182821", strokeWidth=1.5).encode(**enc),
				alt.Chart(KNN_POINTS).mark_point(size=230, filled=True, opacity=1).encode(
					**enc,
					shape=alt.Shape("기호:N", scale=alt.Scale(domain=list(KNN_SHAPES), range=[s for s, _ in KNN_SHAPES.values()]), legend=None),
					color=alt.Color("기호:N", scale=alt.Scale(domain=list(KNN_SHAPES), range=[c for _, c in KNN_SHAPES.values()]), legend=alt.Legend(title=None, orient="top")),
					tooltip=["기호"]),
				alt.Chart(pd.DataFrame({"특성1": [point[0]], "특성2": [point[1]], "t": ["✚"]})).mark_text(fontSize=30, color="#e0457b", fontWeight="bold").encode(**enc, text="t:N"),
			).properties(height=430)
			event = st.altair_chart(chart, width="stretch", on_select="rerun", selection_mode="pick", key="knn_chart")
			picked = (event.get("selection", {}) if event else {}).get("pick") or []
			if picked:
				spot = (round(float(picked[0]["특성1"]), 2), round(float(picked[0]["특성2"]), 2))
				if spot != st.session_state.get("knn_last_pick"):
					st.session_state["knn_last_pick"] = spot
					st.session_state["knn_x"], st.session_state["knn_y"] = spot
					st.rerun()

			table = pd.DataFrame([counts.to_numpy()], columns=[s.split()[0] for s in KNN_SHAPES], index=["개수"])
			res1, res2 = st.columns([0.55, 0.45], gap="medium")
			with res1:
				st.markdown(f"**〈k={k}일 때 분류 결과〉**")
				st.table(table)
			with res2:
				st.metric("분류 결과", winner)
				if len(top) > 1:
					st.caption(f"동점({', '.join(top)})! 가장 가까운 이웃의 기호로 정했어요. 그래서 k는 보통 홀수로 정해요.")
				else:
					st.caption(f"가장 가까운 {k}개 중 {winner.split()[1]}가 {counts.max()}개로 가장 많아요.")
			with st.expander("✚ 위치를 숫자로 정하기"):
				p1, p2 = st.columns(2)
				p1.number_input("특성 1", 0.0, 10.0, step=0.1, key="knn_x")
				p2.number_input("특성 2", 0.0, 10.0, step=0.1, key="knn_y")
			st.markdown('<div class="coach-note">🎬 활용: 철수와 취향이 비슷한 5명(k=5)을 찾아 그들이 재미있게 본 영화를 추천하는 것이 k-NN의 대표적인 활용이에요. '
						'"나와 비슷한 사람들은 무엇을 좋아할까?"</div>', unsafe_allow_html=True)

		with tree_t:
			st.markdown("#### 의사결정트리: 스무고개처럼 질문하며 정답 찾기")
			game, model_part = st.columns([0.5, 0.5], gap="large")
			with game:
				st.markdown("**🐾 동물 스무고개** — 떠올린 동물에 대해 답해 보세요.")
				path = ["q1"]
				q1 = st.radio("날개가 있나요?", ["예", "아니요"], index=None, horizontal=True, key="dt_q1")
				answer = None
				if q1 == "예":
					path.append("q2")
					q2 = st.radio("날 수 있나요?", ["예", "아니요"], index=None, horizontal=True, key="dt_q2")
					answer = {"예": "매", "아니요": "펭귄"}.get(q2)
				elif q1 == "아니요":
					path.append("q3")
					q3 = st.radio("지느러미가 있나요?", ["예", "아니요"], index=None, horizontal=True, key="dt_q3")
					answer = {"예": "돌고래", "아니요": "곰"}.get(q3)
				if answer:
					path.append(answer)
					st.success(f"당신이 떠올린 동물은… **{answer}** 이군요!")
				st.graphviz_chart(animal_tree_dot(path), width="stretch")

				def reset_tree_game():
					for key in ("dt_q1", "dt_q2", "dt_q3"):
						st.session_state.pop(key, None)

				st.button("↺ 다시 하기", on_click=reset_tree_game, key="dt_reset")
			with model_part:
				st.markdown("**🍜 활동: 최종 분류 결과 작성하기**")
				st.caption("트리: 매운 요리 좋아함? → (예) 면 좋아함? 라면/비빔밥 · (아니요) 닭고기 좋아함? 치킨/삼겹살")
				food_rows = [("T", "T", "F", "라면"), ("F", "T", "F", "삼겹살")]
				food_answers = []
				for i, (spicy, noodle, chicken, _) in enumerate(food_rows):
					food_answers.append(st.selectbox(
						f"매운 요리 {spicy} · 면 {noodle} · 닭고기 {chicken} → 음식은?",
						["라면", "비빔밥", "치킨", "삼겹살"], index=None, key=f"dt_food_{i}"))
				if st.button("정답 확인", key="dt_food_check"):
					for (spicy, noodle, chicken, correct), given in zip(food_rows, food_answers):
						route = f"매운 요리 {'예' if spicy == 'T' else '아니요'} → " + (f"면 {'예' if noodle == 'T' else '아니요'}" if spicy == "T" else f"닭고기 {'예' if chicken == 'T' else '아니요'}")
						if given == correct:
							st.success(f"⭕ {correct} ({route})")
						else:
							st.error(f"❌ 다시 생각해 보세요: {route} 를 따라가면?")

			st.divider()
			st.markdown("**🎓 데이터로 트리 학습하기: 대학 합격 예측**")
			adm = admission_data()
			depth = st.slider("트리의 최대 깊이(질문 단계 수)", 1, 4, 2, key="dt_depth")
			tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(adm[["모의고사 점수", "출석률"]], adm["결과"])
			tl, tr = st.columns([0.5, 0.5], gap="medium")
			with tl:
				dot = export_graphviz(tree, feature_names=["모의고사 점수", "출석률"], class_names=list(tree.classes_),
									  filled=True, rounded=True, impurity=False, fontname="Noto Sans KR")
				st.graphviz_chart(dot, width="stretch")
				st.caption(f"학습 정확도 {tree.score(adm[['모의고사 점수', '출석률']], adm['결과']):.0%} · 깊이가 깊을수록 훈련 데이터에 딱 맞지만 과적합 위험이 커져요.")
			with tr:
				my_score = st.slider("내 모의고사 점수", 50, 100, 85, key="dt_score")
				my_attend = st.slider("내 출석률(%)", 70, 100, 92, key="dt_attend")
				verdict = tree.predict(pd.DataFrame({"모의고사 점수": [my_score], "출석률": [my_attend]}))[0]
				st.metric("트리의 예측", "🎉 합격" if verdict == "합격" else "📘 불합격")
				gx, gy = np.meshgrid(np.arange(50, 101, 1.25), np.arange(70, 101, 0.75))
				region = pd.DataFrame({"모의고사 점수": gx.ravel(), "출석률": gy.ravel()})
				region["예측"] = tree.predict(region)
				colors = alt.Scale(domain=["합격", "불합격"], range=["#2f6fb5", "#e0457b"])
				st.altair_chart(alt.layer(
					alt.Chart(region).mark_square(size=40, opacity=0.18).encode(x=alt.X("모의고사 점수:Q", scale=alt.Scale(domain=[50, 100])), y=alt.Y("출석률:Q", scale=alt.Scale(domain=[70, 100])), color=alt.Color("예측:N", scale=colors, legend=None)),
					alt.Chart(adm).mark_circle(size=50).encode(x="모의고사 점수:Q", y="출석률:Q", color=alt.Color("결과:N", scale=colors, legend=alt.Legend(orient="top", title=None))),
					alt.Chart(pd.DataFrame({"모의고사 점수": [my_score], "출석률": [my_attend]})).mark_point(shape="diamond", size=300, filled=True, color="#182821").encode(x="모의고사 점수:Q", y="출석률:Q"),
				).properties(height=300), width="stretch")
				st.caption("배경색 = 트리가 나눈 영역(직사각형 모양) · 점 = 학습 데이터 · ◆ = 나")

		with quiz_t:
			st.markdown("#### 분류 알고리즘 한눈에 정리")
			st.dataframe(pd.DataFrame([
				{"항목": "핵심 아이디어", "로지스틱 회귀": "직선(곡선)으로 나누고 확률로 분류", "SVM": "두 그룹 사이 최대 안전거리(마진)의 경계", "k-NN": "가장 가까운 이웃들의 다수결", "의사결정트리": "질문을 단계적으로 하여 분류"},
				{"항목": "장점", "로지스틱 회귀": "빠르고 해석 쉬움", "SVM": "정확도 높음", "k-NN": "이해하기 쉬움", "의사결정트리": "이해하기 쉽고 설명 가능"},
				{"항목": "단점", "로지스틱 회귀": "복잡한 패턴(비선형)에 약함", "SVM": "데이터가 많으면 느림", "k-NN": "데이터가 많으면 느림", "의사결정트리": "과적합 위험"},
				{"항목": "활용 예시", "로지스틱 회귀": "스팸 메일, 합격 예측", "SVM": "이미지 분류, 얼굴 인식", "k-NN": "추천 시스템, 유사 고객", "의사결정트리": "고객 분석, 의료 진단, 대출 심사"},
			]), hide_index=True, width="stretch")
			st.markdown("#### 🧠 상황에 맞는 알고리즘 고르기")
			algorithms = ["로지스틱 회귀", "서포트 벡터 머신", "k-최근접 이웃", "의사결정트리"]
			with st.form("classify_quiz"):
				picks = [st.radio(f"Q{i + 1}. {question}", algorithms, index=None, horizontal=True, key=f"cq_{i}")
						 for i, (question, _) in enumerate(CLASSIFY_QUIZ)]
				submitted = st.form_submit_button("채점하기", type="primary")
			if submitted:
				score = sum(p == a for p, (_, a) in zip(picks, CLASSIFY_QUIZ))
				st.metric("점수", f"{score} / {len(CLASSIFY_QUIZ)}")
				for i, (pick_value, (_, correct)) in enumerate(zip(picks, CLASSIFY_QUIZ)):
					if pick_value == correct:
						st.write(f"⭕ Q{i + 1}. {correct}")
					else:
						st.write(f"❌ Q{i + 1}. 정답: **{correct}**")
				if score == len(CLASSIFY_QUIZ):
					st.balloons()

	with unsup_tab:
		what_t, vs_t, kmean_t, step_t, init_t, sum_t = st.tabs(
			["① 비지도학습이란?", "② 분류 vs 군집화", "③ K의 의미", "④ k-평균 4단계", "⑤ 처음 중심점의 영향", "⑥ 정리 퀴즈"]
		)

		with what_t:
			st.markdown("#### 비지도학습: 정답 없이 데이터 속 관계와 패턴 찾기")
			st.markdown('<div class="coach-note">비지도학습은 정답(레이블)이 제공되지 않은 데이터를 학습하는 방식이에요. '
						'인공지능 스스로 데이터의 전체적인 특성을 분석해, 비슷한 데이터끼리 묶을 때(군집) 많이 사용합니다.</div>', unsafe_allow_html=True)
			st.markdown("**🧺 직접 해 보기: 레이블이 없는 다섯 가지를 두 군집으로 묶어 보세요.** 기준은 스스로 정해요!")
			pick_cols = st.columns(len(FOOD_ITEMS))
			my_groups = [col.segmented_control(f"{emoji} {name}", ["군집 1", "군집 2"], key=f"food_group_{i}")
						 for i, (col, (emoji, name, _, _)) in enumerate(zip(pick_cols, FOOD_ITEMS))]
			box1, box2 = st.columns(2)
			for box, label in [(box1, "군집 1"), (box2, "군집 2")]:
				members = " ".join(f"{e} {n}" for (e, n, _, _), g in zip(FOOD_ITEMS, my_groups) if g == label)
				box.container(border=True).markdown(f"**[{label}]**  \n{members or '· · ·'}")
			if all(my_groups) and len(set(my_groups)) == 2:
				def same_split(attr_index):
					keys = [item[attr_index] for item in FOOD_ITEMS]
					pairs = {(g, k) for g, k in zip(my_groups, keys)}
					return len(pairs) == 2 and len({g for g, _ in pairs}) == 2
				if same_split(2):
					st.success("🍎 **종류(과일 / 채소)** 를 기준으로 묶었네요!")
				elif same_split(3):
					st.success("📏 **모양(길쭉한 / 동그란)** 을 기준으로 묶었네요!")
				else:
					st.info("🤔 나만의 기준으로 묶었네요! 어떤 특성으로 묶었는지 친구에게 설명해 보세요. (예: 색깔)")
				st.caption("정답은 하나가 아니에요. 같은 데이터라도 어떤 특성을 보느냐에 따라 군집이 달라집니다 — 그래서 '정답 없는' 학습이에요.")
			with st.expander("교과서의 두 가지 군집화 보기"):
				e1, e2 = st.columns(2)
				for col, title, idx in [(e1, "❶ 종류별로 군집화", 2), (e2, "❷ 비슷한 모양끼리 군집화", 3)]:
					col.markdown(f"**{title}**")
					for value in dict.fromkeys(item[idx] for item in FOOD_ITEMS):
						col.write(f"[{value}] " + " ".join(e for e, *_ in [it for it in FOOD_ITEMS if it[idx] == value]))

		with vs_t:
			st.markdown("#### 분류와 군집화는 무엇이 다를까?")
			st.dataframe(pd.DataFrame([
				{"구분": "학습 방식", "분류": "지도학습", "군집화": "비지도학습"},
				{"구분": "데이터", "분류": "레이블(정답)이 있음", "군집화": "레이블이 없음"},
				{"구분": "하는 일", "분류": "새 데이터가 미리 정한 그룹 중 어디에 속하는지 판별", "군집화": "비슷한 특성을 가진 몇 개의 그룹으로 나눔"},
				{"구분": "목적", "분류": "그룹을 맞히는 것", "군집화": "군집화 결과로 얻은 정보를 활용하는 것"},
			]), hide_index=True, width="stretch")
			st.markdown("**🧠 어느 쪽일까요?**")
			with st.form("vs_quiz"):
				vs_picks = [st.radio(f"Q{i + 1}. {q}", ["분류", "군집화"], index=None, horizontal=True, key=f"vsq_{i}")
							for i, (q, _) in enumerate(CLUSTER_OR_CLASSIFY)]
				vs_done = st.form_submit_button("채점하기", type="primary")
			if vs_done:
				vs_score = sum(p == a for p, (_, a) in zip(vs_picks, CLUSTER_OR_CLASSIFY))
				st.metric("점수", f"{vs_score} / {len(CLUSTER_OR_CLASSIFY)}")
				for i, (p, (_, a)) in enumerate(zip(vs_picks, CLUSTER_OR_CLASSIFY)):
					st.write(f"{'⭕' if p == a else '❌'} Q{i + 1}. 정답: **{a}** — " + ("정답(레이블)이 있어요." if a == "분류" else "정답 없이 비슷한 것끼리 묶어요."))
				if vs_score == len(CLUSTER_OR_CLASSIFY):
					st.balloons()

		with kmean_t:
			st.markdown("#### K = 몇 개의 그룹(모둠)으로 나눌까?")
			st.markdown('<div class="coach-note">우리 반 친구 36명의 키와 성적 데이터예요. K를 바꾸면 k-평균이 비슷한 친구끼리 자동으로 모둠을 만들어 줍니다.</div>', unsafe_allow_html=True)
			mates = classmates().copy()
			group_k = st.slider("K (모둠 수)", 1, 6, 2, key="mates_k")
			scaled = (mates - mates.mean()) / mates.std()
			model = KMeans(n_clusters=group_k, n_init=10, random_state=0).fit(scaled)
			mates["모둠"] = [f"모둠 {l + 1}" for l in model.labels_]
			centers = mates.groupby("모둠")[["키", "성적"]].mean().reset_index()
			palette = alt.Scale(domain=[f"모둠 {i + 1}" for i in range(6)], range=CLUSTER_COLORS)
			g1, g2 = st.columns([0.62, 0.38], gap="medium")
			with g1:
				st.altair_chart(alt.layer(
					alt.Chart(mates).mark_circle(size=110, opacity=0.85).encode(
						x=alt.X("키:Q", scale=alt.Scale(domain=[145, 190]), title="키 (cm)"), y=alt.Y("성적:Q", scale=alt.Scale(domain=[40, 100])),
						color=alt.Color("모둠:N", scale=palette, legend=alt.Legend(orient="top", title=None)), tooltip=["키", "성적", "모둠"]),
					alt.Chart(centers).mark_text(text="★", fontSize=34, stroke="#182821", strokeWidth=0.8).encode(
						x="키:Q", y="성적:Q", color=alt.Color("모둠:N", scale=palette, legend=None)),
				).properties(height=380), width="stretch")
			with g2:
				summary = mates.groupby("모둠").agg(인원=("키", "size"), 평균키=("키", "mean"), 평균성적=("성적", "mean")).round(1)
				st.dataframe(summary, width="stretch")
				st.caption("★ = 각 모둠의 중심점(평균 위치)")
				st.markdown('<div class="coach-note">모둠마다 어떤 특징이 있나요? 예: "키가 크고 성적이 높은 모둠". '
							'군집에 의미를 붙이는 것은 사람의 몫이에요.</div>', unsafe_allow_html=True)

		with step_t:
			st.markdown("#### k-평균 알고리즘 동작 과정 (4단계 반복)")
			st.caption("학급에서 반장을 뽑는다고 생각해 보세요. 👩‍🎓 학생들 = 데이터, ★ 반장 = 중심점")
			km = st.session_state.get("km")
			sc1, sc2 = st.columns([0.3, 0.7], gap="medium")
			with sc1:
				km_k = st.slider("군집 수 K", 2, 6, 3, key="km_k")
				if km is None or (km["k"] != km_k and not st.session_state.get("km_custom")):
					kmeans_reset(km_k, km["seed"] if km else 0)
					km = st.session_state["km"]
				custom_mode = st.toggle("🖱️ 중심점을 클릭해서 직접 놓기", key="km_custom_mode")
				if custom_mode:
					custom = st.session_state.setdefault("km_custom", [])
					st.info(f"그래프를 클릭해 중심점 {km_k}개를 놓으세요. ({len(custom)}/{km_k})")
					st.button("▶ 놓은 중심점으로 시작", on_click=kmeans_use_custom, disabled=len(custom) != km_k, type="primary", width="stretch", key="km_use_custom")
					st.button("지우기", on_click=lambda: st.session_state.update(km_custom=[]), width="stretch", key="km_clear_custom")
				else:
					next_label = "2단계: 가까운 팀으로" if km["phase"] == "assign" else "3단계: 중심점 이동"
					st.button(f"▶ {next_label}", on_click=kmeans_step, disabled=km["done"], width="stretch", type="primary", key="km_step")
					st.button("⏩ 끝까지 실행", on_click=kmeans_finish, disabled=km["done"], width="stretch", key="km_finish")
					st.button("🎲 1단계: 중심점 다시 뽑기", on_click=kmeans_reset, args=(km_k, km["seed"] + 1), width="stretch", key="km_reseed")
				st.metric("반복 횟수", km["step"])

			if km["labels"] is None:
				stage = 0
			elif km["done"]:
				stage = 3
			else:
				stage = 1 if km["phase"] == "move" else 2
			with sc2:
				customers = customer_blobs().copy()
				xs, xa = axis("특성 1 (방문 빈도)")
				ys, ya = axis("특성 2 (구매 금액)")
				names = [f"{chr(65 + i)}팀" for i in range(6)]
				palette = alt.Scale(domain=names + ["소속 없음"], range=CLUSTER_COLORS + ["#9aa39a"])
				customers["팀"] = [names[l] for l in km["labels"]] if km["labels"] is not None else "소속 없음"
				cdf = pd.DataFrame(km["centers"], columns=["방문 빈도", "구매 금액"])
				cdf["팀"] = names[:len(cdf)]
				enc = dict(x=alt.X("방문 빈도:Q", scale=xs, axis=xa), y=alt.Y("구매 금액:Q", scale=ys, axis=ya))
				grid = pd.DataFrame([(gx, gy) for gx in np.arange(0, 10.01, 0.25) for gy in np.arange(0, 10.01, 0.25)], columns=["방문 빈도", "구매 금액"])
				place = alt.selection_point(name="place", fields=["방문 빈도", "구매 금액"], on="click", nearest=True)
				layers = [alt.Chart(grid).mark_point(opacity=0.001, size=60).encode(**enc).add_params(place)]
				if km["labels"] is not None and stage == 1:
					links = customers.join(cdf.set_index("팀"), on="팀", rsuffix="_c")
					layers.append(alt.Chart(links).mark_rule(opacity=0.35).encode(
						**enc, x2="방문 빈도_c:Q", y2="구매 금액_c:Q", color=alt.Color("팀:N", scale=palette, legend=None)))
				layers.append(alt.Chart(customers).mark_circle(size=55, opacity=0.85).encode(
					**enc, color=alt.Color("팀:N", scale=palette, legend=alt.Legend(orient="top", title=None))))
				if km["prev"] is not None:
					prev = pd.DataFrame(km["prev"], columns=["방문 빈도", "구매 금액"])
					prev["팀"] = names[:len(prev)]
					moves = prev.join(cdf.set_index("팀"), on="팀", rsuffix="_n")
					layers.append(alt.Chart(moves).mark_rule(strokeWidth=2.5, strokeDash=[4, 3]).encode(
						**enc, x2="방문 빈도_n:Q", y2="구매 금액_n:Q", color=alt.Color("팀:N", scale=palette, legend=None)))
					layers.append(alt.Chart(prev).mark_text(text="☆", fontSize=30).encode(**enc, color=alt.Color("팀:N", scale=palette, legend=None)))
				if not custom_mode:
					layers.append(alt.Chart(cdf).mark_text(text="★", fontSize=38, stroke="#182821", strokeWidth=0.8).encode(
						**enc, color=alt.Color("팀:N", scale=palette, legend=None)))
				else:
					placed = pd.DataFrame(st.session_state.get("km_custom", []), columns=["방문 빈도", "구매 금액"])
					placed["번호"] = [str(i + 1) for i in range(len(placed))]
					layers.append(alt.Chart(placed).mark_text(text="★", fontSize=38, color="#182821").encode(**enc))
				event = st.altair_chart(alt.layer(*layers).properties(height=420), width="stretch", on_select="rerun", selection_mode="place", key="km_chart")
				picked = (event.get("selection", {}) if event else {}).get("place") or []
				if custom_mode and picked:
					spot = (round(float(picked[0]["방문 빈도"]), 2), round(float(picked[0]["구매 금액"]), 2))
					custom = st.session_state.setdefault("km_custom", [])
					if spot != st.session_state.get("km_last_place") and len(custom) < km_k:
						st.session_state["km_last_place"] = spot
						custom.append(list(spot))
						st.rerun()

			stage_cols = st.columns(4, gap="small")
			for i, (col, (badge, title, desc)) in enumerate(zip(stage_cols, KMEANS_STAGES)):
				with col.container(border=True):
					mark = "👉 " if i == stage and not custom_mode else ""
					st.markdown(f"{mark}**{badge}** {title}" if i != stage else f":red[{mark}**{badge}** {title}]")
					st.caption(desc)
			if km["done"]:
				st.success(f"🎉 중심점이 더 이상 이동하지 않아요. {km['step']}번 반복으로 군집화 완료!")
				data = customer_blobs().to_numpy()
				inertia = float(((data - np.array(km["centers"])[km["labels"]]) ** 2).sum())
				st.caption(f"각 데이터와 중심점 사이 거리 제곱의 합: {inertia:,.1f} (작을수록 촘촘하게 묶임)")
			elif stage == 1:
				st.caption("선 = 각 데이터가 가장 가까운 중심점(반장)의 팀에 소속된 모습. 데이터가 실제로 움직이는 게 아니라 팀에 소속되는 거예요!")
			elif stage == 2:
				st.caption("☆ → ★ : 팀 데이터들의 평균(가운데) 위치로 중심점이 이동했어요.")

		with init_t:
			st.markdown("#### 처음 중심점에 따라 결과가 달라질 수 있어요")
			st.markdown('<div class="coach-note">같은 데이터, 같은 K=4로 처음 중심점만 다르게 뽑아 세 번 군집화했어요. '
						'어떤 결과가 가장 잘 묶였나요? 🎲를 눌러 여러 번 실험해 보세요.</div>', unsafe_allow_html=True)
			st.session_state.setdefault("init_seed", 0)
			if st.button("🎲 처음 중심점 다시 뽑기", key="init_reroll"):
				st.session_state["init_seed"] += 3
			blobs = four_blobs()
			runs = []
			for offset in range(3):
				seed = st.session_state["init_seed"] + offset
				model = KMeans(n_clusters=4, init="random", n_init=1, max_iter=100, random_state=seed).fit(blobs)
				runs.append((seed, model))
			best = min(model.inertia_ for _, model in runs)
			xs, xa = axis("특성 1")
			ys, ya = axis("특성 2")
			palette = alt.Scale(domain=[f"군집 {i + 1}" for i in range(4)], range=CLUSTER_COLORS[:4])
			for col, (seed, model) in zip(st.columns(3, gap="small"), runs):
				frame = blobs.assign(군집=[f"군집 {l + 1}" for l in model.labels_])
				cents = pd.DataFrame(model.cluster_centers_, columns=["특성1", "특성2"]).assign(군집=[f"군집 {i + 1}" for i in range(4)])
				with col:
					st.altair_chart(alt.layer(
						alt.Chart(frame).mark_circle(size=22).encode(x=alt.X("특성1:Q", scale=xs, axis=xa), y=alt.Y("특성2:Q", scale=ys, axis=ya), color=alt.Color("군집:N", scale=palette, legend=None)),
						alt.Chart(cents).mark_text(text="★", fontSize=26, stroke="#182821", strokeWidth=0.6).encode(x="특성1:Q", y="특성2:Q", color=alt.Color("군집:N", scale=palette, legend=None)),
					).properties(height=240), width="stretch")
					good = model.inertia_ <= best * 1.01
					st.markdown(f"{'🏆 ' if good else ''}시도 {seed + 1} · 거리 제곱합 **{model.inertia_:,.0f}**")
			st.caption("거리 제곱합이 작을수록 데이터가 중심점 가까이 잘 모여 있어요. 그래서 실제로는 여러 번 시작해 가장 좋은 결과를 고릅니다.")

		with sum_t:
			st.markdown("#### 용어 정리")
			t1, t2, t3 = st.columns(3, gap="small")
			t1.container(border=True).markdown("**⚪ 데이터(점)**  \n분석할 대상 하나하나")
			t2.container(border=True).markdown("**🔴🔵 군집(클러스터)**  \n비슷한 데이터들이 모여 만들어진 그룹")
			t3.container(border=True).markdown("**★ 중심점(센트로이드)**  \n군집의 대표 위치(평균 위치)")
			p1, p2 = st.columns(2, gap="medium")
			p1.container(border=True).markdown("**👍 장점**  \n- 알고리즘이 간단해서 이해하기 쉬워요.\n- 계산이 빠르고 큰 데이터도 처리할 수 있어요.\n- 다양한 분야에서 널리 사용돼요.")
			p2.container(border=True).markdown("**👎 단점**  \n- 처음 중심점을 어떻게 정하느냐에 따라 결과가 달라질 수 있어요.\n- 그룹(모둠) 개수 K를 미리 정해야 해요.")
			st.markdown('<div class="coach-note">🎯 핵심: k-평균 알고리즘은 "가까운 데이터끼리 묶고, 평균 위치로 대표를 이동하는 것"을 반복해서 그룹을 만드는 방법이에요!</div>', unsafe_allow_html=True)
			st.markdown("#### 🧠 확인 퀴즈")
			with st.form("kmeans_quiz"):
				km_picks = [st.radio(f"Q{i + 1}. {q}", options, index=None, key=f"kmq_{i}") for i, (q, options, _) in enumerate(KMEANS_QUIZ)]
				km_done = st.form_submit_button("채점하기", type="primary")
			if km_done:
				km_score = sum(p == a for p, (_, _, a) in zip(km_picks, KMEANS_QUIZ))
				st.metric("점수", f"{km_score} / {len(KMEANS_QUIZ)}")
				for i, (p, (_, _, a)) in enumerate(zip(km_picks, KMEANS_QUIZ)):
					st.write(f"{'⭕' if p == a else '❌'} Q{i + 1}. 정답: **{a}**")
				if km_score == len(KMEANS_QUIZ):
					st.balloons()


with tabs[0]:
	st.markdown('<div class="section-label">CHOOSE THE LEARNING TASK</div>', unsafe_allow_html=True)
	st.subheader("문제의 목표가 학습 방법을 결정해요")
	scenario = st.selectbox(
		"해결하고 싶은 문제를 선택하세요",
		["집값 예측", "스팸 메일 판별", "비슷한 소비 습관의 고객 찾기", "직접 문제 정하기"],
	)
	recommendations = {
		"집값 예측": ("지도학습 · 회귀", "집값처럼 연속된 숫자를 예측합니다.", "선형 회귀로 관계를 살펴보거나, 결정 트리 회귀로 복잡한 규칙을 찾을 수 있어요."),
		"스팸 메일 판별": ("지도학습 · 분류", "메일에 스팸/정상이라는 정답(레이블)이 있습니다.", "로지스틱 회귀는 기준을 세워 분류하고, 결정 트리는 판단 규칙을 따라 분류합니다."),
		"비슷한 소비 습관의 고객 찾기": ("비지도학습 · 군집화", "정답 레이블 없이 소비 패턴이 비슷한 데이터를 묶습니다.", "K-평균은 정한 군집 수만큼 중심을 이동시키며 가까운 데이터끼리 묶습니다."),
		"직접 문제 정하기": ("목표와 데이터부터 확인", "정답이 있는지, 결과가 숫자인지 범주인지 먼저 살펴보세요.", "정답이 있고 숫자를 예측하면 회귀, 범주를 맞히면 분류, 정답 없이 묶으면 군집화입니다."),
	}
	kind, reason, algorithm = recommendations[scenario]
	left, right = st.columns([1, 1.2], gap="large")
	with left:
		st.metric("추천 유형", kind)
		st.write(reason)
	with right:
		st.markdown("#### 알고리즘 후보")
		st.write(algorithm)
		st.markdown(
			'<div class="coach-note">생각해 보기: 입력 데이터가 달라지거나 정답 레이블이 사라지면, '
			'지금 선택한 학습 유형도 그대로 적절할까요?</div>',
			unsafe_allow_html=True,
		)

	st.divider()
	st.markdown("#### 한눈에 비교하기")
	comparison = pd.DataFrame(
		[
			{"학습 유형": "지도학습", "데이터에 필요한 것": "입력값과 정답 레이블", "주요 과제": "회귀 · 분류", "예시 알고리즘": "선형 회귀 · 결정 트리"},
			{"학습 유형": "비지도학습", "데이터에 필요한 것": "입력값 (정답 레이블 없음)", "주요 과제": "군집화", "예시 알고리즘": "K-평균"},
		]
	)
	st.dataframe(comparison, hide_index=True, width="stretch")
	with st.expander("훈련 데이터와 테스트 데이터는 왜 나눌까요?"):
		st.write(
			"훈련 데이터는 모델이 규칙과 패턴을 배우는 데 사용합니다. 테스트 데이터는 학습 중에는 보여 주지 않고, "
			"새로운 데이터에도 잘 작동하는지 마지막에 확인하는 데 사용합니다. 같은 데이터를 훈련과 평가에 모두 쓰면 "
			"실제보다 성능이 좋아 보일 수 있어요."
		)

with tabs[1]:
	st.markdown('<div class="section-label">BUILD / TEST / REFLECT</div>', unsafe_allow_html=True)
	st.subheader("나만의 데이터로 모델 생성하기")
	st.write("기본 데이터로 먼저 실험하거나, 정답 열이 포함된 CSV를 올려 직접 모델을 만들어 보세요.")
	source = st.selectbox(
		"실습 데이터",
		["붓꽃 분류 데이터", "당뇨 진행도 데이터", "CSV 업로드", "가상 고객 데이터 (군집화)"],
		index=3,
		help="기본 선택은 군집화와 실루엣 계수 실습입니다. 다른 데이터로 바꾸어 분류·회귀도 실험할 수 있어요.",
	)

	uploaded = None
	if source == "붓꽃 분류 데이터":
		dataset = load_iris(as_frame=True)
		data = dataset.data.copy()
		data["정답"] = dataset.target.map(dict(enumerate(dataset.target_names)))
		target_column = "정답"
		task = "분류"
	elif source == "당뇨 진행도 데이터":
		dataset = load_diabetes(as_frame=True)
		data = dataset.data.copy()
		data["진행도"] = dataset.target
		target_column = "진행도"
		task = "회귀"
	elif source == "가상 고객 데이터 (군집화)":
		features, _ = make_blobs(n_samples=240, centers=4, cluster_std=1.25, random_state=17)
		data = pd.DataFrame(features, columns=["방문 빈도", "구매 금액"])
		target_column = None
		task = "군집화"
	else:
		uploaded = st.file_uploader("데이터 CSV 파일", type=["csv"], key="ml_csv_upload")
		if uploaded is None:
			st.info("CSV를 올린 뒤 분류 또는 회귀를 선택하세요. 첫 행은 열 이름이어야 합니다.")
			data = None
			target_column = None
			task = "분류"
		else:
			try:
				data = pd.read_csv(uploaded)
				if data.empty or data.columns.empty:
					raise ValueError("데이터 행과 열이 있는 CSV 파일을 선택해 주세요.")
				target_column = st.selectbox("정답(레이블) 열", data.columns.tolist(), key="ml_target_column")
				task = st.radio("예측할 결과", ["분류", "회귀"], horizontal=True, key="ml_custom_task")
			except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
				st.error(f"CSV를 읽지 못했어요: {error}")
				data = None
				target_column = None
				task = "분류"

	if data is not None:
		with st.expander("현재 데이터 미리 보기"):
			st.dataframe(data.head(12), hide_index=True, width="stretch")
			st.caption(f"{len(data)}개 행 · {len(data.columns)}개 열")

		if task == "분류":
			algorithm = st.selectbox("분류 알고리즘", ["로지스틱 회귀", "결정 트리", "퍼셉트론"])
			test_percent = st.slider("테스트 데이터 비율", 20, 40, 25, 5, format="%d%%", key="classification_test_size")
		elif task == "회귀":
			algorithm = st.selectbox("회귀 알고리즘", ["선형 회귀", "결정 트리 회귀"])
			test_percent = st.slider("테스트 데이터 비율", 20, 40, 25, 5, format="%d%%", key="regression_test_size")
		else:
			algorithm = "K-평균"
			cluster_count = st.slider("찾을 군집 수 (K)", 2, 8, 4)
			st.caption("군집화는 정답 레이블 없이 전체 데이터를 학습합니다.")

		result_signature = (
			source,
			task,
			target_column,
			algorithm,
			test_percent if task != "군집화" else cluster_count,
			(uploaded.name, uploaded.size) if uploaded is not None else None,
		)
		train_button_label = "K-평균 군집화하고 실루엣 계수 보기" if task == "군집화" else "모델 훈련하고 평가하기"
		if st.button(train_button_label, type="primary", icon=":material/model_training:"):
			try:
				clean_data = data.dropna(subset=[target_column]).copy() if target_column else data.copy()
				if len(clean_data) < 10:
					raise ValueError("학습과 평가를 위해 최소 10개 행이 필요합니다.")
				if task == "군집화":
					features = pd.get_dummies(clean_data, dummy_na=True).astype(float).fillna(0)
					if features.shape[1] < 1:
						raise ValueError("군집화에 사용할 숫자 또는 범주형 열이 없습니다.")
					model = make_pipeline(StandardScaler(), KMeans(n_clusters=cluster_count, n_init=10, random_state=42))
					labels = model.fit_predict(features)
					cluster_model = model.named_steps["kmeans"]
					scaled_features = model.named_steps["standardscaler"].transform(features)
					unique_labels = pd.Series(labels).nunique()
					average_silhouette = (
						silhouette_score(scaled_features, labels)
						if 2 <= unique_labels < len(labels)
						else None
					)
					result = {
						"task": task,
						"labels": labels,
						"features": features,
						"scaled_features": pd.DataFrame(scaled_features, index=features.index, columns=features.columns),
						"inertia": cluster_model.inertia_,
						"silhouette": average_silhouette,
						"k": cluster_count,
					}
				else:
					features = clean_data.drop(columns=[target_column])
					features = pd.get_dummies(features, dummy_na=True).astype(float).fillna(0)
					if features.shape[1] < 1:
						raise ValueError("정답 열 외에 학습할 특성 열이 필요합니다.")
					labels = clean_data[target_column]
					if task == "분류":
						labels = LabelEncoder().fit_transform(labels.astype(str))
						estimator = {
							"로지스틱 회귀": LogisticRegression(max_iter=1000),
							"결정 트리": DecisionTreeClassifier(max_depth=5, random_state=42),
							"퍼셉트론": Perceptron(max_iter=1000, random_state=42),
						}[algorithm]
						stratify = labels if pd.Series(labels).value_counts().min() >= 2 else None
					else:
						labels = pd.to_numeric(labels, errors="coerce")
						valid = labels.notna()
						features, labels = features.loc[valid], labels.loc[valid]
						estimator = {"선형 회귀": LinearRegression(), "결정 트리 회귀": DecisionTreeRegressor(max_depth=5, random_state=42)}[algorithm]
						stratify = None
					x_train, x_test, y_train, y_test = train_test_split(
						features, labels, test_size=test_percent / 100, random_state=42, stratify=stratify
					)
					model = make_pipeline(StandardScaler(), estimator) if algorithm in ["로지스틱 회귀", "퍼셉트론", "선형 회귀"] else estimator
					model.fit(x_train, y_train)
					predictions = model.predict(x_test)
					if task == "분류":
						result = {"task": task, "accuracy": accuracy_score(y_test, predictions), "matrix": confusion_matrix(y_test, predictions), "train": len(x_train), "test": len(x_test)}
					else:
						result = {"task": task, "mae": mean_absolute_error(y_test, predictions), "r2": r2_score(y_test, predictions), "train": len(x_train), "test": len(x_test)}
				st.session_state["ml_lab_result"] = result
				st.session_state["ml_lab_result_signature"] = result_signature
			except (ValueError, TypeError) as error:
				st.error(f"모델을 만들지 못했어요: {error}")

		result = st.session_state.get("ml_lab_result")
		if result and st.session_state.get("ml_lab_result_signature") == result_signature:
			st.divider()
			if result["task"] == "분류":
				metric_cols = st.columns(3)
				metric_cols[0].metric("정확도", f"{result['accuracy']:.1%}")
				metric_cols[1].metric("훈련 데이터", f"{result['train']}개")
				metric_cols[2].metric("테스트 데이터", f"{result['test']}개")
				st.markdown("#### 혼동 행렬")
				st.caption("행은 실제 정답, 열은 모델의 예측입니다. 대각선 값이 클수록 정답과 예측이 일치합니다.")
				st.dataframe(pd.DataFrame(result["matrix"]), width="stretch")
			elif result["task"] == "회귀":
				metric_cols = st.columns(4)
				metric_cols[0].metric("평균 절대 오차 (MAE)", f"{result['mae']:.2f}")
				metric_cols[1].metric("결정 계수 (R²)", f"{result['r2']:.2f}")
				metric_cols[2].metric("훈련 데이터", f"{result['train']}개")
				metric_cols[3].metric("테스트 데이터", f"{result['test']}개")
				st.caption("MAE는 예측이 평균적으로 얼마나 빗나갔는지, R²는 기준 모델보다 얼마나 설명력이 있는지 나타냅니다.")
			else:
				metric_cols = st.columns(3)
				metric_cols[0].metric("군집 수", f"{result['k']}개")
				metric_cols[1].metric("군집 내 제곱 거리 합", f"{result['inertia']:.1f}")
				if result["silhouette"] is not None:
					metric_cols[2].metric("평균 실루엣 계수", f"{result['silhouette']:.3f}")
				else:
					metric_cols[2].metric("평균 실루엣 계수", "계산 불가")
					st.warning("실루엣 계수는 실제로 만들어진 군집이 2개 이상일 때 계산할 수 있습니다.")

				st.markdown("#### 데이터 점 하나의 실루엣 계수 계산")
				st.write(
					"점 i가 속한 군집 안에서 다른 점까지의 평균 거리 a(i), "
					"가장 가까운 다른 군집까지의 평균 거리 b(i)를 비교합니다. 거리는 모델 학습과 같은 표준화 특성 공간에서 계산합니다."
				)
				point_index = st.number_input(
					"계산할 데이터 점 i",
					min_value=0,
					max_value=len(result["labels"]) - 1,
					value=0,
					step=1,
				)
				point_index = int(point_index)
				labels = result["labels"]
				point_label = labels[point_index]
				distances = pairwise_distances(
					result["scaled_features"].iloc[[point_index]], result["scaled_features"]
				)[0]
				own_cluster_indices = [index for index, label in enumerate(labels) if label == point_label and index != point_index]
				a_distance = float(distances[own_cluster_indices].mean()) if own_cluster_indices else 0.0
				other_cluster_distances = []
				for label in sorted(set(labels)):
					if label == point_label:
						continue
					cluster_indices = [index for index, other_label in enumerate(labels) if other_label == label]
					other_cluster_distances.append(
						{"가까운 군집": f"군집 {chr(65 + int(label))}", "i에서 군집까지 평균 거리": float(distances[cluster_indices].mean())}
					)
				b_distance = min(item["i에서 군집까지 평균 거리"] for item in other_cluster_distances) if other_cluster_distances else 0.0
				silhouette = (b_distance - a_distance) / max(a_distance, b_distance) if max(a_distance, b_distance) else 0.0

				chart_data = result["features"].iloc[:, :2].copy()
				chart_data["군집"] = [f"군집 {chr(65 + int(label))}" for label in labels]
				chart_data["선택 점 크기"] = [180 if index == point_index else 50 for index in range(len(labels))]
				if chart_data.shape[1] >= 2:
					st.scatter_chart(chart_data, x=chart_data.columns[0], y=chart_data.columns[1], color="군집", size="선택 점 크기")
				else:
					st.info("군집 분포 그래프를 그리려면 숫자 또는 범주형 특성이 두 개 이상 필요합니다.")
				st.dataframe(pd.DataFrame(other_cluster_distances), hide_index=True, width="stretch")
				measure_cols = st.columns(3)
				measure_cols[0].metric("군집 내 평균 거리 a(i)", f"{a_distance:.3f}")
				measure_cols[1].metric("가까운 군집 평균 거리 b(i)", f"{b_distance:.3f}")
				measure_cols[2].metric("실루엣 계수 s(i)", f"{silhouette:.3f}")
				st.latex(r"s(i) = \frac{b(i) - a(i)}{\max(a(i),\,b(i))}")
				if not own_cluster_indices:
					st.caption("이 점은 군집 안에 혼자 있어 a(i)를 0으로 두었습니다.")
				st.caption("s(i)가 1에 가까우면 현재 군집에 잘 속하고, 0에 가까우면 경계에 있으며, 음수이면 다른 군집에 더 가까울 수 있습니다.")
				st.caption("군집 내 제곱 거리 합은 작을수록 각 점이 중심에 가깝지만, K가 커지면 자연히 작아집니다.")

			with st.expander("결과를 해석할 때 확인할 점"):
				st.write(
					"테스트 성능 하나만으로 좋은 모델이라고 단정하지 마세요. 데이터의 양과 균형, "
					"특성 선택, 훈련/테스트 분할 방식에 따라 결과가 달라집니다. 특히 실제 생활에 적용할 때는 "
					"잘못된 예측이 누구에게 어떤 영향을 주는지도 함께 살펴야 합니다."
				)

with tabs[2]:
	st.markdown('<div class="section-label">FROM A SINGLE NEURON TO DEEP LEARNING</div>', unsafe_allow_html=True)
	st.subheader("퍼셉트론: 인공 신경망의 출발점")
	st.write("퍼셉트론은 입력값에 가중치를 곱해 더한 뒤, 기준을 넘는지 판단하는 가장 단순한 인공 뉴런입니다.")
	logic_gate = st.selectbox("학습할 논리 문제", ["AND", "OR", "XOR"])
	gate_data = {
		"AND": ([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1]),
		"OR": ([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 1]),
		"XOR": ([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0]),
	}
	gate_x, gate_y = gate_data[logic_gate]
	gate_model = Perceptron(max_iter=1000, random_state=7).fit(gate_x, gate_y)
	gate_predictions = gate_model.predict(gate_x)
	gate_frame = pd.DataFrame(gate_x, columns=["입력 A", "입력 B"])
	gate_frame["정답"] = gate_y
	gate_frame["퍼셉트론 예측"] = gate_predictions
	gate_left, gate_right = st.columns([1, 1], gap="large")
	with gate_left:
		st.dataframe(gate_frame, hide_index=True, width="stretch")
	with gate_right:
		st.metric("훈련 데이터 정답률", f"{accuracy_score(gate_y, gate_predictions):.0%}")
		if logic_gate == "XOR":
			st.info("XOR는 한 개의 직선으로 두 클래스를 나눌 수 없어 단층 퍼셉트론만으로 해결할 수 없습니다. 은닉층을 추가하면 어떤 변화가 생길까요?")
		else:
			st.caption("AND와 OR는 한 개의 직선으로 나눌 수 있는 문제라 단층 퍼셉트론으로 학습할 수 있습니다.")

	st.divider()
	st.subheader("심층 신경망과 딥러닝 직접 비교하기")
	st.write("은닉층의 수와 뉴런 수를 바꾸며, 굽은 경계가 있는 데이터를 얼마나 잘 분류하는지 살펴보세요.")
	controls = st.columns(3)
	with controls[0]:
		architecture = st.selectbox("은닉층 구조", ["뉴런 2개", "뉴런 8개", "은닉층 2개 (8, 8)", "은닉층 3개 (16, 8, 4)"])
	with controls[1]:
		activation = st.selectbox("활성화 함수", ["relu", "tanh"])
	with controls[2]:
		noise = st.slider("데이터 잡음", 0.0, 0.45, 0.20, 0.05)

	layer_options = {"뉴런 2개": (2,), "뉴런 8개": (8,), "은닉층 2개 (8, 8)": (8, 8), "은닉층 3개 (16, 8, 4)": (16, 8, 4)}
	neural_signature = (architecture, activation, noise)
	if st.button("신경망 학습하고 결과 보기", type="primary", icon=":material/psychology:"):
		moon_x, moon_y = make_moons(n_samples=400, noise=noise, random_state=21)
		x_train, x_test, y_train, y_test = train_test_split(moon_x, moon_y, test_size=0.3, random_state=42, stratify=moon_y)
		network = make_pipeline(
			StandardScaler(),
			MLPClassifier(hidden_layer_sizes=layer_options[architecture], activation=activation, max_iter=1500, early_stopping=True, random_state=42),
		)
		network.fit(x_train, y_train)
		moon_predictions = network.predict(x_test)
		st.session_state["neural_result"] = {
			"accuracy": accuracy_score(y_test, moon_predictions),
			"x": x_test,
			"y": y_test,
			"predictions": moon_predictions,
			"architecture": architecture,
			"noise": noise,
			"signature": neural_signature,
		}

	neural_result = st.session_state.get("neural_result")
	if neural_result and neural_result.get("signature") == neural_signature:
		result_cols = st.columns(3)
		result_cols[0].metric("테스트 정확도", f"{neural_result['accuracy']:.1%}")
		result_cols[1].metric("구조", neural_result["architecture"])
		result_cols[2].metric("데이터 잡음", f"{neural_result['noise']:.2f}")
		scatter = pd.DataFrame(neural_result["x"], columns=["특성 1", "특성 2"])
		scatter["모델 예측"] = [f"클래스 {label}" for label in neural_result["predictions"]]
		st.scatter_chart(scatter, x="특성 1", y="특성 2", color="모델 예측")
		st.caption("그래프는 학습에 사용하지 않은 테스트 데이터입니다. 구조와 잡음을 바꾸어 정확도와 예측 결과를 비교해 보세요.")

	with st.expander("인공 신경망과 딥러닝"):
		st.write(
			"인공 신경망은 입력층, 하나 이상의 은닉층, 출력층의 뉴런이 연결된 모델입니다. "
			"은닉층이 여러 개인 신경망을 깊은 신경망(Deep Neural Network)이라 하며, "
			"이 신경망을 데이터로 학습하는 방법을 딥러닝이라고 부릅니다. 층이 깊다고 언제나 더 좋은 것은 아니며, "
			"데이터의 양과 문제에 맞는 구조인지 함께 살펴야 합니다."
		)
