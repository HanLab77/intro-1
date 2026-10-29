import streamlit as st


# 모든 페이지에서 메뉴(큰 탭)와 서브메뉴(탭 안의 탭)가 한눈에 구분되도록 단계별로 다르게 꾸민다.
#   1단계 메뉴: 흰 막대 위의 큰 버튼, 선택하면 진한 초록
#   2단계 서브메뉴: "세부 메뉴" 표시가 붙은 작은 알약 버튼, 선택하면 연두, 내용은 흰 카드 안에
#   3단계 메뉴: 밑줄만 있는 작은 글자 탭
MENU_CSS = """
<style>
/* ── 사이드바: 페이지 이동 메뉴와 페이지 안의 보조 메뉴 구분 ── */
[data-testid="stSidebarNavItems"] {
	background: #fff;
	border: 1px solid #dce2d9;
	border-radius: 12px;
	padding: .45rem;
	box-shadow: 0 2px 6px rgba(35, 70, 55, .08);
}
[data-testid="stSidebarNavItems"]::before {
	content: "MENU · 페이지 이동";
	display: block;
	padding: .15rem .55rem .4rem;
	color: #68766e;
	font: 600 .68rem 'DM Mono', monospace;
	letter-spacing: .08em;
}
[data-testid="stSidebarNavLink"] { border-radius: 8px; }
[data-testid="stSidebarNavLink"] span { font-weight: 600; }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: #234637; }
[data-testid="stSidebarNavLink"][aria-current="page"] span { color: #fff; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
	border-left: 3px solid #95b55a;
	padding: .2rem 0 .2rem .75rem;
}

/* ── 탭 공통: 기본 밑줄 표시는 숨기고 단계별 모양을 쓴다 ── */
[data-testid="stTabs"] .react-aria-SelectionIndicator { display: none; }
[data-testid="stTabs"] [role="tab"] p { margin: 0; color: inherit; }

/* ── 1단계 메뉴 ── */
[role="tablist"]:not([data-testid="stTabPanel"] [role="tablist"]) {
	gap: .4rem;
	flex-wrap: wrap;
	height: auto;
	padding: .45rem;
	background: #fff;
	border: 1px solid #dce2d9;
	border-radius: 14px;
	box-shadow: 0 2px 8px rgba(35, 70, 55, .08);
}
[role="tablist"]:not([data-testid="stTabPanel"] [role="tablist"]) > [role="tab"] {
	height: auto;
	padding: .6rem 1.1rem;
	border-radius: 10px;
	color: #234637;
	transition: all .15s ease;
}
[role="tablist"]:not([data-testid="stTabPanel"] [role="tablist"]) > [role="tab"] p { font-size: 1.02rem; font-weight: 700; }
[role="tablist"]:not([data-testid="stTabPanel"] [role="tablist"]) > [role="tab"][data-hovered] { background: #dce8d9; }
[role="tablist"]:not([data-testid="stTabPanel"] [role="tablist"]) > [role="tab"][data-selected] {
	background: #234637;
	color: #fff;
	box-shadow: 0 3px 10px rgba(35, 70, 55, .25);
}
[data-testid="stTabPanel"] { padding-top: 1rem; }

/* ── 2단계 서브메뉴 ── */
[data-testid="stTabPanel"] [role="tablist"] {
	gap: .4rem;
	flex-wrap: wrap;
	align-items: center;
	height: auto;
	margin-top: .8rem;
	padding: 0;
	border: none;
}
[data-testid="stTabPanel"] [role="tablist"]::before {
	content: "세부 메뉴";
	margin-right: .3rem;
	padding: .2rem .55rem;
	border-radius: 4px;
	background: #234637;
	color: #d6ee76;
	font-size: .72rem;
	font-weight: 700;
}
[data-testid="stTabPanel"] [role="tablist"] > [role="tab"] {
	height: auto;
	padding: .3rem .85rem;
	border: 1px solid #cfdcc8;
	border-radius: 999px;
	background: #f1f5ec;
	color: #3b5a4a;
	transition: all .15s ease;
}
[data-testid="stTabPanel"] [role="tablist"] > [role="tab"] p { font-size: .9rem; font-weight: 600; }
[data-testid="stTabPanel"] [role="tablist"] > [role="tab"][data-hovered] { background: #e4eddc; }
[data-testid="stTabPanel"] [role="tablist"] > [role="tab"][data-selected] {
	background: #d6ee76;
	border-color: #a9c84a;
	color: #234637;
}
[data-testid="stTabPanel"] [data-testid="stTabPanel"] {
	margin-top: .5rem;
	padding: 1.1rem 1.3rem;
	background: #fff;
	border: 1px solid #dce2d9;
	border-left: 4px solid #95b55a;
	border-radius: 12px;
}

/* ── 3단계 메뉴: 서브메뉴 카드 안의 밑줄 탭 ── */
[data-testid="stTabPanel"] [data-testid="stTabPanel"] [role="tablist"] {
	gap: .1rem;
	margin-top: .4rem;
	border-bottom: 1px solid #dce2d9;
}
[data-testid="stTabPanel"] [data-testid="stTabPanel"] [role="tablist"]::before { content: none; }
[data-testid="stTabPanel"] [data-testid="stTabPanel"] [role="tablist"] > [role="tab"] {
	padding: .35rem .7rem;
	border: none;
	border-bottom: 2px solid transparent;
	border-radius: 0;
	background: transparent;
	color: #68766e;
}
[data-testid="stTabPanel"] [data-testid="stTabPanel"] [role="tablist"] > [role="tab"] p { font-size: .85rem; font-weight: 500; }
[data-testid="stTabPanel"] [data-testid="stTabPanel"] [role="tablist"] > [role="tab"][data-hovered] { background: #f1f5ec; }
[data-testid="stTabPanel"] [data-testid="stTabPanel"] [role="tablist"] > [role="tab"][data-selected] {
	border-bottom-color: #234637;
	background: transparent;
	color: #234637;
}
[data-testid="stTabPanel"] [data-testid="stTabPanel"] [role="tablist"] > [role="tab"][data-selected] p { font-weight: 700; }
[data-testid="stTabPanel"] [data-testid="stTabPanel"] [data-testid="stTabPanel"] {
	margin-top: 0;
	padding: 1rem 0 0;
	background: transparent;
	border: none;
}
</style>
"""


def apply_menu_style():
	st.markdown(MENU_CSS, unsafe_allow_html=True)
