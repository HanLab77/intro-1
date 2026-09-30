import pandas as pd
import streamlit as st
from sklearn.datasets import load_diabetes, load_iris, make_blobs, make_moons
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, pairwise_distances, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


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

st.write("")
tabs = st.tabs(["01 · 학습 유형과 알고리즘", "02 · 모델 만들기와 평가", "03 · 퍼셉트론과 딥러닝"])

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
