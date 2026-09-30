import ast
import re
import subprocess
import sys

import streamlit as st


st.set_page_config(
	page_title="Python 코드 실험실",
	page_icon="⌘",
	layout="wide",
)

st.markdown(
	"""
	<style>
	@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');
	:root {
		--ink: #182821;
		--forest: #234637;
		--lime: #d6ee76;
		--paper: #f6f7f1;
		--muted: #68766e;
		--line: #dce2d9;
	}
	html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
	.stApp { background: var(--paper); color: var(--ink); }
	.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }
	.lab-header {
		background: var(--forest); color: #f6f7f1; padding: 2rem 2.4rem;
		border-radius: 8px; margin-bottom: 1.5rem;
	}
	.lab-kicker { color: var(--lime); font: 500 .74rem 'DM Mono', monospace; }
	.lab-header h1 { color: #fff; font-size: 2rem; margin: .5rem 0; }
	.lab-header p { color: #dbe6dc; margin: 0; line-height: 1.7; }
	.panel-title { color: var(--forest); font-size: 1.1rem; font-weight: 700; }
	.coach-note { border-left: 3px solid #95b55a; padding: .2rem 0 .2rem 1rem; line-height: 1.8; }
	div[data-testid="stForm"] { background: #fff; border-color: var(--line); }
	div[data-testid="stTextArea"] textarea { font-family: 'DM Mono', monospace; }
	@media (max-width: 700px) {
		.lab-header { padding: 1.5rem; }
		.lab-header h1 { font-size: 1.65rem; }
		.block-container { padding-top: 1rem; }
	}
	</style>
	""",
	unsafe_allow_html=True,
)


EXAMPLES = {
	"직접 작성하기": "print('안녕하세요, 파이썬!')",
	"변수와 계산": "name = '파이썬'\nminutes = 25\nprint(f'{name} 공부를 {minutes}분 했어요.')",
	"조건문": "score = 82\n\nif score >= 80:\n    print('통과')\nelse:\n    print('다시 도전')",
	"반복문": "for number in range(1, 6):\n    print(number, '번째 도전')",
}


def get_advice(source, output, error):
	if error:
		if "SyntaxError" in error:
			return "문장 구조를 확인해 보세요. 괄호와 따옴표가 짝을 이루는지, 콜론(:)이 필요한 줄에 있는지 살펴보세요."
		if "IndentationError" in error:
			return "들여쓰기 간격이 맞지 않습니다. 같은 코드 블록 안의 줄은 공백 4칸으로 통일해 보세요."
		if "ZeroDivisionError" in error:
			return "0으로 나누려 해서 실행이 멈췄어요. 나누는 값이 0인지 먼저 확인하는 조건을 추가해 보세요."
		if "NameError" in error:
			match = re.search(r"name '([^']+)' is not defined", error)
			name = f"`{match.group(1)}`" if match else "변수"
			return f"{name} 이름을 찾을 수 없습니다. 철자가 같은지, 사용하기 전에 값을 정했는지 확인해 보세요."
		if "TypeError" in error:
			return "서로 다른 자료형을 함께 사용했을 수 있어요. `type()`으로 각 값의 자료형을 확인해 보세요."
		if "ValueError" in error:
			return "값의 형식이 기대한 것과 다릅니다. 변환하려는 문자열이나 입력값을 확인해 보세요."
		return "오류 메시지의 마지막 줄과 표시된 코드 줄을 먼저 살펴보세요. 작은 부분으로 나누어 다시 실행하면 원인을 찾기 쉽습니다."

	if not output.strip():
		return "실행은 끝났지만 화면에 출력된 값이 없어요. 확인하고 싶은 변수에 `print()`를 사용해 보세요."
	if "for " in source or "while " in source:
		return "반복이 잘 실행됐어요. 반복 횟수나 범위를 바꿔 결과가 어떻게 달라지는지 실험해 보세요."
	if "if " in source:
		return "조건에 따른 결과를 확인했어요. 조건값을 참과 거짓이 되도록 각각 바꿔 두 경로를 비교해 보세요."
	if "input(" in source:
		return "입력값을 사용했어요. 입력값을 바꾸어 실행하고, 예상하지 못한 입력도 어떻게 처리할지 생각해 보세요."
	return "코드가 오류 없이 실행됐어요. 변수의 값을 바꿔 다시 실행하거나, 계산 과정을 한 단계씩 출력해 결과를 확인해 보세요."


def run_python(source, input_text):
	runner = r'''import ast
import contextlib
import io
import sys
import traceback

source = sys.argv[1]
input_lines = iter(sys.argv[2].splitlines())
output_limit = 20000

class LimitedOutput(io.StringIO):
	def write(self, text):
		remaining = output_limit - self.tell()
		if remaining > 0:
			super().write(text[:remaining])
		if len(text) > remaining and "출력 제한" not in self.getvalue():
			super().write("\\n... 출력 제한 (20,000자)\\n")
		return len(text)

def read_input(prompt=""):
	if prompt:
		print(prompt, end="")
	try:
		return next(input_lines)
	except StopIteration:
		raise EOFError("입력값이 부족합니다. 아래 입력값 칸에 값을 추가하세요.")

safe_builtins = {
	"print": print, "input": read_input, "range": range, "len": len,
	"int": int, "float": float, "str": str, "bool": bool,
	"list": list, "tuple": tuple, "dict": dict, "set": set,
	"sum": sum, "min": min, "max": max, "abs": abs, "round": round,
	"enumerate": enumerate, "zip": zip, "sorted": sorted, "reversed": reversed,
	"type": type, "isinstance": isinstance, "ValueError": ValueError,
	"TypeError": TypeError, "IndexError": IndexError, "KeyError": KeyError,
	"ZeroDivisionError": ZeroDivisionError, "Exception": Exception,
}
output = LimitedOutput()
error = ""
try:
	tree = ast.parse(source, filename="학생 코드")
	for node in ast.walk(tree):
		if isinstance(node, (ast.Import, ast.ImportFrom)):
			raise ValueError("이 실습 환경에서는 import를 사용할 수 없습니다.")
		if isinstance(node, ast.Name) and node.id.startswith("__"):
			raise ValueError("내부 전용 이름은 사용할 수 없습니다.")
		if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
			raise ValueError("내부 전용 속성은 사용할 수 없습니다.")
	with contextlib.redirect_stdout(output):
		exec(compile(tree, "학생 코드", "exec"), {"__builtins__": safe_builtins})
except BaseException as exception:
	error = f"{type(exception).__name__}: {exception}"
	if isinstance(exception, SyntaxError) and exception.lineno:
		error += f" (줄 {exception.lineno})"
sys.stdout.write(output.getvalue())
sys.stderr.write(error)
'''
	try:
		result = subprocess.run(
			[sys.executable, "-I", "-c", runner, source, input_text],
			capture_output=True,
			text=True,
			timeout=3,
			check=False,
		)
	except subprocess.TimeoutExpired:
		return "", "실행 시간이 3초를 넘었습니다. 무한 반복이나 너무 큰 반복 범위가 있는지 확인해 보세요."

	error = result.stderr.strip()
	if result.returncode != 0 and not error:
		error = "코드 실행 프로세스가 정상적으로 끝나지 않았습니다."
	return result.stdout, error


def load_example():
	st.session_state["python_lab_code"] = EXAMPLES[
		st.session_state["python_lab_example"]
	]


st.markdown(
	'<header class="lab-header"><div class="lab-kicker">JAUN HIGH SCHOOL / PYTHON LAB</div>'
	'<h1>Python 코드 실험실</h1>'
	'<p>코드를 직접 실행하고, 결과와 오류를 바탕으로 다음 실험의 힌트를 받아보세요.</p></header>',
	unsafe_allow_html=True,
)

editor_col, result_col = st.columns([1.08, 0.92], gap="large")
with editor_col:
	st.markdown('<div class="panel-title">코드 편집기</div>', unsafe_allow_html=True)
	st.selectbox(
		"시작 예제",
		list(EXAMPLES),
		key="python_lab_example",
		on_change=load_example,
	)
	if "python_lab_code" not in st.session_state:
		st.session_state["python_lab_code"] = EXAMPLES["직접 작성하기"]
	with st.form("python_runner_form"):
		source = st.text_area(
			"파이썬 코드",
			height=330,
			label_visibility="collapsed",
			placeholder="실행할 Python 코드를 입력하세요.",
			key="python_lab_code",
		)
		input_text = st.text_area(
			"input() 입력값",
			height=90,
			placeholder="input()을 여러 번 사용하면 한 줄에 하나씩 입력하세요.",
		)
		submitted = st.form_submit_button(
			"코드 실행",
			type="primary",
			icon=":material/play_arrow:",
			use_container_width=True,
		)

with result_col:
	st.markdown('<div class="panel-title">실행 결과</div>', unsafe_allow_html=True)
	if submitted:
		output, error = run_python(source, input_text)
		st.session_state["python_lab_result"] = {
			"output": output,
			"error": error,
			"source": source,
		}

	result = st.session_state.get("python_lab_result")
	if result:
		if result["error"]:
			st.error(result["error"])
		else:
			st.success("실행이 완료됐습니다.")
		st.code(result["output"] or "출력된 결과가 없습니다.", language="text")

		st.markdown('<div class="panel-title">코드 코치</div>', unsafe_allow_html=True)
		st.markdown(
			'<div class="coach-note">'
			+ get_advice(result["source"], result["output"], result["error"])
			+ "</div>",
			unsafe_allow_html=True,
		)
	else:
		st.info("코드를 실행하면 결과와 맞춤 조언이 여기에 표시됩니다.")

st.divider()
st.caption(
	"코드 코치는 코드와 오류 메시지를 분석하는 규칙 기반 피드백이며 외부 AI 서비스에 연결되지 않습니다. "
	"안전한 학습을 위해 import와 내부 속성 접근이 제한되고, 실행 시간은 3초로 제한됩니다."
)
