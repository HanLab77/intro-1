import ast
import os
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
		--ink: #1e293b;
		--blue: #2563eb;
		--paper: #f3f4f8;
		--muted: #64748b;
		--line: #e2e8f0;
	}
	html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
	.stApp { background: var(--paper); color: var(--ink); }
	header[data-testid="stHeader"] { display: none; }
	.block-container { max-width: 1440px; padding-top: 1.2rem; padding-bottom: 2.5rem; }
	.panel-title { color: #1d4ed8; font-size: 1.12rem; font-weight: 700; margin-bottom: .6rem; }
	.learning-hero {
		background: linear-gradient(110deg, #3478ed, #1d4ed8); color: #fff;
		border: 1px solid rgba(255, 255, 255, .35); border-radius: 18px;
		padding: 2.2rem 1.4rem; text-align: center; margin-bottom: .5rem;
	}
	.learning-hero h1 { color: #fff; font-size: 2.5rem; line-height: 1.3; margin: 0 0 .45rem; text-shadow: 0 2px 5px rgba(0, 0, 0, .25); }
	.learning-hero p { color: #eff6ff; font-size: 1.25rem; line-height: 1.6; margin: 0; }
	.api-status { border: 2px solid #22c55e; border-radius: 14px; background: #dcfce7; color: #078344; text-align: center; padding: 1rem 1.2rem; }
	.api-status h2 { color: inherit; font-size: 1.3rem; margin: 0 0 .35rem; }
	.api-status p { color: inherit; font-size: 1.05rem; line-height: 1.6; margin: 0; }
	.api-status.unconfigured { border-color: #f59e0b; background: #fff7d6; color: #8a5700; }
	div[data-testid="stVerticalBlockBorderWrapper"] {
		background: #fff; border: 1px solid rgba(15, 23, 42, .04);
		border-radius: 12px; box-shadow: 0 8px 24px rgba(15, 23, 42, .09);
	}
	.st-key-challenge-list { max-height: 270px; overflow-y: auto; padding-right: .2rem; }
	.st-key-challenge-list button {
		min-height: 72px; height: auto; padding: .7rem .9rem; border-radius: 10px;
		border-left: 3px solid #3b82f6; text-align: left; white-space: pre-wrap;
		background: linear-gradient(135deg, #f8fafc, #e2e8f0); color: #1e3a8a;
	}
	.st-key-challenge-list button p { text-align: left; line-height: 1.45; }
	.st-key-editor textarea { font-family: 'DM Mono', monospace; font-size: .9rem; }
	.st-key-result-panel { min-height: 680px; }
	.st-key-result-output pre {
		box-sizing: border-box; height: 250px; overflow: auto;
		background: #1f2937 !important; color: #f9fafb !important;
		border: 1px solid #334155; border-radius: 8px;
	}
	.st-key-result-output pre code { color: #f9fafb !important; }
	.coach-note { border-left: 3px solid #10b981; padding: .2rem 0 .2rem 1rem; line-height: 1.8; }
	div[data-testid="stForm"] { border: 0; padding: 0; background: transparent; }
	div[data-testid="stTextArea"] textarea { font-family: 'DM Mono', monospace; }
	@media (max-width: 700px) {
		.block-container { padding: .7rem .6rem 2rem; }
		.learning-hero { padding: 1.5rem .9rem; }
		.learning-hero h1 { font-size: 1.55rem; }
		.learning-hero p { font-size: 1rem; }
		.api-status { padding: .85rem .7rem; }
		.st-key-result-panel { min-height: 0; }
		.st-key-result-output pre { height: 220px; }
	}
	</style>
	""",
	unsafe_allow_html=True,
)

api_key_configured = bool(os.getenv("OPENAI_API_KEY"))
if not api_key_configured:
	try:
		api_key_configured = bool(st.secrets.get("OPENAI_API_KEY"))
	except Exception:
		api_key_configured = False

st.markdown(
	'<section class="learning-hero">'
	'<h1>🐍 자운고 파이썬 프로그래밍 학습환경(VER 1.0)</h1>'
	'<p>단계별 학습과 실시간 피드백으로 파이썬을 마스터하세요!</p>'
	'</section>',
	unsafe_allow_html=True,
)
if api_key_configured:
	api_status_title = "🔑 API 연결됨"
	api_status_message = "OpenAI API가 설정되어 있습니다. AI 튜터와 함께 학습하세요!"
	api_status_class = "api-status"
else:
	api_status_title = "🔑 API 미설정"
	api_status_message = "OPENAI_API_KEY를 설정하면 AI 튜터 기능을 사용할 수 있습니다."
	api_status_class = "api-status unconfigured"

st.markdown(
	f'<section class="{api_status_class}"><h2>{api_status_title}</h2>'
	f'<p>{api_status_message}</p></section>',
	unsafe_allow_html=True,
)


CHALLENGES = [
	{
		"title": "1단계: Hello World",
		"description": "첫 번째 파이썬 프로그램으로 인사말을 출력해보세요.",
		"code": "print('Hello, Python!')",
		"hint": "print() 괄호 안에 출력하고 싶은 문장을 따옴표로 감싸 적어보세요.",
	},
	{
		"title": "2단계: 변수와 데이터 타입",
		"description": "문자열, 정수, 실수 변수를 만들고 한 줄씩 출력해보세요.",
		"code": "name = '홍길동'\nage = 17\nheight = 170.5\nprint(f'이름: {name}, 나이: {age}, 키: {height}cm')",
		"hint": "문자열은 따옴표로, 정수와 실수는 숫자로 값을 저장한 뒤 f-string으로 출력해보세요.",
	},
	{
		"title": "3단계: 리스트 다루기",
		"description": "과일 이름들을 담은 리스트를 만들고 반복문으로 출력해보세요.",
		"code": "fruits = ['사과', '바나나', '오렌지', '포도']\nfor fruit in fruits:\n    print(f'좋아하는 과일: {fruit}')",
		"hint": "for fruit in fruits: 형태로 리스트를 순회하고, 반복문 안의 print 줄은 공백 4칸 들여쓰세요.",
	},
	{
		"title": "4단계: 조건문",
		"description": "숫자를 입력받아 짝수인지 홀수인지 판단해보세요.",
		"code": "number = 10\n\nif number % 2 == 0:\n    print(f'{number}는 짝수입니다')\nelse:\n    print(f'{number}는 홀수입니다')",
		"hint": "숫자를 2로 나눈 나머지가 0이면 짝수입니다. if와 else 뒤에 콜론을 붙여보세요.",
	},
	{
		"title": "5단계: 반복문 (for)",
		"description": "1부터 10까지 출력하고, 숫자의 합도 계산해보세요.",
		"code": "total = 0\nfor number in range(1, 11):\n    print(number)\n    total += number\nprint(f'1부터 10까지의 합: {total}')",
		"hint": "range(1, 11)은 1부터 10까지 만듭니다. total에 각 숫자를 더해 누적해보세요.",
	},
	{
		"title": "6단계: 함수 만들기",
		"description": "두 숫자를 더하는 함수를 만들고 결과를 출력해보세요.",
		"code": "def add_numbers(first, second):\n    return first + second\n\nresult = add_numbers(5, 3)\nprint(f'5 + 3 = {result}')",
		"hint": "함수 안에서 두 값을 더해 return하고, 함수를 호출한 결과를 변수에 담아 출력해보세요.",
	},
	{
		"title": "7단계: 딕셔너리",
		"description": "학생 정보를 딕셔너리에 저장하고 항목을 출력해보세요.",
		"code": "student = {'이름': '민지', '학년': 2, '과목': '정보'}\nfor key, value in student.items():\n    print(f'{key}: {value}')",
		"hint": "딕셔너리의 key와 value를 함께 순회하려면 .items()를 사용할 수 있어요.",
	},
	{
		"title": "8단계: 예외 처리",
		"description": "0으로 나누는 오류를 try-except로 처리해보세요.",
		"code": "try:\n    result = 10 / 0\n    print(result)\nexcept ZeroDivisionError:\n    print('0으로 나눌 수 없습니다!')",
		"hint": "오류가 발생할 수 있는 코드를 try 안에 두고, ZeroDivisionError를 except에서 처리해보세요.",
	},
	{
		"title": "9단계: 클래스 만들기",
		"description": "간단한 Dog 클래스를 만들고 객체를 사용해보세요.",
		"code": "class Dog:\n    def __init__(self, name):\n        self.name = name\n\n    def bark(self):\n        print(f'{self.name}: 멍멍!')\n\nmy_dog = Dog('초코')\nmy_dog.bark()",
		"hint": "클래스의 __init__에서 이름을 저장하고, 인스턴스 메서드에서 self.name을 사용해보세요.",
	},
]


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
	"__build_class__": __build_class__,
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
		exec(compile(tree, "학생 코드", "exec"), {"__builtins__": safe_builtins, "__name__": "__main__"})
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


def select_challenge(index):
	challenge = CHALLENGES[index]
	st.session_state["python_lab_selected"] = index
	st.session_state["python_lab_code"] = challenge["code"]
	st.session_state.pop("python_lab_result", None)
	st.session_state.pop("python_lab_help", None)


if "python_lab_selected" not in st.session_state:
	st.session_state["python_lab_selected"] = 2
if "python_lab_code" not in st.session_state:
	st.session_state["python_lab_code"] = CHALLENGES[2]["code"]

editor_col, result_col = st.columns([1, 1], gap="large")
with editor_col:
	with st.container(border=True):
		st.markdown('<div class="panel-title">🎯 도전과제</div>', unsafe_allow_html=True)
		with st.container(height=270, border=False, key="challenge-list"):
			for index, challenge in enumerate(CHALLENGES):
				label = f"**{challenge['title']}**  \n{challenge['description']}"
				st.button(
					label,
					key=f"challenge_{index}",
					on_click=select_challenge,
					args=(index,),
					type="primary" if index == st.session_state["python_lab_selected"] else "secondary",
					use_container_width=True,
				)

	with st.container(border=True, key="editor"):
		st.markdown('<div class="panel-title">💻 코드 편집기</div>', unsafe_allow_html=True)
		with st.form("python_runner_form"):
			source = st.text_area(
				"파이썬 코드",
				height=165,
				label_visibility="collapsed",
				placeholder="실행할 Python 코드를 입력하세요.",
				key="python_lab_code",
			)
			input_text = st.text_area(
				"input() 입력값",
				height=75,
				placeholder="input()을 사용하는 경우 입력값을 한 줄씩 적으세요.",
			)
			run_col, help_col = st.columns([1, 1], gap="small")
			with run_col:
				run_clicked = st.form_submit_button(
					"▶ 실행하기",
					type="primary",
					use_container_width=True,
				)
			with help_col:
				help_clicked = st.form_submit_button(
					"💡 도움받기",
					use_container_width=True,
				)

selected = CHALLENGES[st.session_state["python_lab_selected"]]
if run_clicked:
	output, error = run_python(source, input_text)
	st.session_state["python_lab_result"] = {
		"output": output,
		"error": error,
		"source": source,
	}
	st.session_state.pop("python_lab_help", None)
if help_clicked:
	st.session_state["python_lab_help"] = selected["hint"]

with result_col:
	with st.container(border=True, key="result-panel"):
		st.markdown('<div class="panel-title">📊 실행 결과</div>', unsafe_allow_html=True)
		result = st.session_state.get("python_lab_result")
		if result and result["error"]:
			st.error(result["error"])
		elif result:
			st.success("실행이 완료됐습니다.")

		with st.container(key="result-output"):
			if result:
				result_text = result["output"] or "출력된 결과가 없습니다."
			else:
				result_text = f"{selected['title']} 코드가 로드되었습니다.\n실행 버튼을 눌러보세요!"
			st.code(result_text, language="text")

		help_text = st.session_state.get("python_lab_help")
		if help_text:
			st.markdown('<div class="panel-title">💡 도움말</div>', unsafe_allow_html=True)
			st.markdown(f'<div class="coach-note">{help_text}</div>', unsafe_allow_html=True)
		elif result:
			st.markdown('<div class="panel-title">🧭 코드 코치</div>', unsafe_allow_html=True)
			st.markdown(
				'<div class="coach-note">'
				+ get_advice(result["source"], result["output"], result["error"])
				+ "</div>",
				unsafe_allow_html=True,
			)

st.caption(
	"코드 코치는 규칙 기반 피드백을 제공합니다. 안전한 학습을 위해 import와 내부 속성 접근이 제한되고, "
	"코드 실행 시간은 3초로 제한됩니다."
)
