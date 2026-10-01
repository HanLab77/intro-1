import json
from datetime import datetime
from pathlib import Path
import secrets
import uuid

import streamlit as st


st.set_page_config(page_title="자유게시판 | 자운고 정보교실", page_icon="📌", layout="wide")

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "freeboard_posts.json"
ADMIN_NUMBER = "1234"

st.markdown(
	"""
	<style>
	@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');
	:root { --ink: #182821; --forest: #234637; --lime: #d6ee76; --paper: #f6f7f1; --muted: #68766e; --line: #dce2d9; }
	html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
	.stApp { background: var(--paper); color: var(--ink); }
	[data-testid="stSidebar"] { background: #edf1e9; border-right: 1px solid var(--line); }
	.block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 4rem; }
	.board-hero { background: var(--forest); color: #f6f7f1; padding: 2.5rem 2.8rem; border-radius: 8px; position: relative; overflow: hidden; animation: arrive .5s ease-out both; }
	.board-hero:after { content: ''; position: absolute; width: 220px; height: 220px; right: -55px; top: -95px; border: 1px solid rgba(214,238,118,.45); border-radius: 50%; box-shadow: 0 0 0 25px rgba(214,238,118,.07), 0 0 0 50px rgba(214,238,118,.05); }
	.board-kicker { color: var(--lime); font: 500 .74rem 'DM Mono', monospace; }
	.board-hero h1 { color: #fff; font-size: 2.5rem; line-height: 1.3; margin: .65rem 0; }
	.board-hero p { color: #dbe6dc; margin: 0; line-height: 1.8; }
	.post-number { color: var(--muted); font: 500 .72rem 'DM Mono', monospace; }
	.stMetric { background: #fff; border: 1px solid var(--line); padding: .8rem 1rem; border-radius: 6px; }
	div[data-testid="stForm"] { border-color: var(--line); background: #fff; }
	@keyframes arrive { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
	@media (max-width: 700px) { .board-hero { padding: 1.8rem 1.3rem; } .board-hero h1 { font-size: 2rem; } .block-container { padding-top: 1.2rem; } }
	</style>
	""",
	unsafe_allow_html=True,
)


def load_posts():
	if not DATA_FILE.exists():
		return []
	try:
		with DATA_FILE.open(encoding="utf-8") as file:
			posts = json.load(file)
		return posts if isinstance(posts, list) else []
	except (OSError, json.JSONDecodeError):
		st.error("게시글 데이터를 읽지 못했습니다. 데이터 파일을 확인해 주세요.")
		return []


def save_posts(posts):
	DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
	temporary_file = DATA_FILE.with_suffix(".tmp")
	with temporary_file.open("w", encoding="utf-8") as file:
		json.dump(posts, file, ensure_ascii=False, indent=2)
	temporary_file.replace(DATA_FILE)


posts = load_posts()
posts.sort(key=lambda post: post.get("created_at", ""), reverse=True)
liked_posts = st.session_state.setdefault("liked_posts", set())

st.markdown(
	'<section class="board-hero"><div class="board-kicker">JA WOON HIGH SCHOOL / COMMUNITY</div>'
	'<h1>자유게시판</h1><p>수업 이야기부터 작은 질문까지, 편하게 나누어 보세요.</p></section>',
	unsafe_allow_html=True,
)
st.write("")

summary_columns = st.columns([1, 1, 2])
summary_columns[0].metric("전체 게시글", f"{len(posts)}")
summary_columns[1].metric("오늘의 날짜", datetime.now().strftime("%Y.%m.%d"))
with summary_columns[2]:
	search_query = st.text_input("게시글 검색", placeholder="제목, 내용, 작성자 검색", label_visibility="collapsed")

left_column, right_column = st.columns([1.5, 1], gap="large")
with left_column:
	st.subheader("게시글")
	if search_query.strip():
		query = search_query.strip().casefold()
		visible_posts = [
			post for post in posts
			if query in post.get("title", "").casefold()
			or query in post.get("content", "").casefold()
			or query in post.get("author", "").casefold()
		]
	else:
		visible_posts = posts

	if not visible_posts:
		st.info("아직 게시글이 없습니다. 첫 이야기를 남겨 주세요." if not search_query else "검색 결과가 없습니다.")
	for post in visible_posts:
		with st.container(border=True):
			st.markdown(f'<div class="post-number">POST / {post.get("created_at", "")[:10]}</div>', unsafe_allow_html=True)
			st.markdown(f'#### {post.get("title", "제목 없음")}')
			st.caption(f'{post.get("author", "익명")} · {post.get("created_at", "")[:16].replace("T", " ")}')
			st.write(post.get("content", ""))
			post_id = post["id"]
			post.setdefault("likes", 0)
			post.setdefault("comments", [])
			like_label = "좋아요 취소" if post_id in liked_posts else "좋아요"
			if st.button(f'{like_label} · {post["likes"]}', key=f"like_{post_id}", icon=":material/thumb_up:"):
				if post_id in liked_posts:
					liked_posts.remove(post_id)
					post["likes"] = max(0, post["likes"] - 1)
				else:
					liked_posts.add(post_id)
					post["likes"] += 1
				save_posts(posts)
				st.rerun()
			with st.expander(f'댓글 {len(post["comments"])}개'):
				for comment in post["comments"]:
					st.markdown(f'**{comment.get("author", "익명")}** · {comment.get("created_at", "")[:16].replace("T", " ")}')
					st.write(comment.get("content", ""))
				with st.form(f"comment_form_{post_id}", clear_on_submit=True):
					comment_author = st.text_input("댓글 작성자", key=f"comment_author_{post_id}", max_chars=20, placeholder="이름 또는 별명")
					comment_content = st.text_area("댓글 내용", key=f"comment_content_{post_id}", max_chars=1000, placeholder="댓글을 남겨 주세요.")
					comment_submitted = st.form_submit_button("댓글 등록")
				if comment_submitted:
					if not comment_author.strip() or not comment_content.strip():
						st.warning("댓글 작성자와 내용을 입력해 주세요.")
					else:
						post["comments"].append({
							"id": uuid.uuid4().hex,
							"author": comment_author.strip(),
							"content": comment_content.strip(),
							"created_at": datetime.now().isoformat(timespec="minutes"),
						})
						save_posts(posts)
						st.success("댓글을 등록했습니다.")
						st.rerun()

with right_column:
	st.subheader("새 글 쓰기")
	with st.form("new_post_form", clear_on_submit=True):
		author = st.text_input("작성자", max_chars=20, placeholder="이름 또는 별명")
		title = st.text_input("제목", max_chars=80, placeholder="게시글 제목")
		content = st.text_area("내용", height=180, max_chars=3000, placeholder="함께 나누고 싶은 이야기를 적어 주세요.")
		submitted = st.form_submit_button("게시글 등록", type="primary", use_container_width=True)
	if submitted:
		if not author.strip() or not title.strip() or not content.strip():
			st.warning("작성자, 제목, 내용을 모두 입력해 주세요.")
		else:
			posts.append({
				"id": uuid.uuid4().hex,
				"author": author.strip(),
				"title": title.strip(),
				"content": content.strip(),
				"created_at": datetime.now().isoformat(timespec="minutes"),
				"likes": 0,
				"comments": [],
			})
			save_posts(posts)
			st.success("게시글을 등록했습니다.")
			st.rerun()

	st.divider()
	with st.expander("관리자 메뉴"):
		if not st.session_state.get("board_admin", False):
			with st.form("admin_login_form"):
				admin_number = st.text_input("관리번호", type="password", max_chars=20)
				login_submitted = st.form_submit_button("관리자 로그인")
			if login_submitted:
				if secrets.compare_digest(admin_number, ADMIN_NUMBER):
					st.session_state.board_admin = True
					st.rerun()
				st.error("관리번호가 올바르지 않습니다.")
		else:
			st.success("관리자 모드")
			if st.button("관리자 로그아웃", key="admin_logout"):
				st.session_state.board_admin = False
				st.rerun()
			if posts:
				selected_id = st.selectbox(
					"관리할 게시글",
					[post["id"] for post in posts],
					format_func=lambda post_id: next(
						f'{post.get("title", "제목 없음")} · {post.get("author", "익명")}'
						for post in posts if post["id"] == post_id
					),
				)
				selected_post = next(post for post in posts if post["id"] == selected_id)
				with st.form("edit_post_form"):
					edited_title = st.text_input("제목 수정", value=selected_post.get("title", ""), max_chars=80)
					edited_author = st.text_input("작성자 수정", value=selected_post.get("author", ""), max_chars=20)
					edited_content = st.text_area("내용 수정", value=selected_post.get("content", ""), height=150, max_chars=3000)
					edit_submitted = st.form_submit_button("수정 저장", type="primary", use_container_width=True)
				if edit_submitted:
					if not edited_title.strip() or not edited_author.strip() or not edited_content.strip():
						st.warning("작성자, 제목, 내용을 비워 둘 수 없습니다.")
					else:
						selected_post.update({
							"title": edited_title.strip(),
							"author": edited_author.strip(),
							"content": edited_content.strip(),
						})
						save_posts(posts)
						st.success("게시글을 수정했습니다.")
						st.rerun()
				with st.form("delete_post_form"):
					confirm_delete = st.checkbox("이 게시글을 영구 삭제합니다.")
					delete_submitted = st.form_submit_button("게시글 삭제", type="secondary", use_container_width=True)
				if delete_submitted:
					if confirm_delete:
						save_posts([post for post in posts if post["id"] != selected_id])
						st.success("게시글을 삭제했습니다.")
						st.rerun()
					st.warning("삭제하려면 확인 항목을 선택해 주세요.")
			else:
				st.caption("관리할 게시글이 없습니다.")

			comment_entries = [
				(post, comment)
				for post in posts
				for comment in post.get("comments", [])
			]
			st.divider()
			st.markdown("#### 댓글 관리")
			if comment_entries:
				comment_ids = [comment["id"] for _, comment in comment_entries]
				selected_comment_id = st.selectbox(
					"관리할 댓글",
					comment_ids,
					format_func=lambda comment_id: next(
						f'{post.get("title", "제목 없음")} / {comment.get("author", "익명")}: '
						f'{comment.get("content", "")[:35]}'
						for post, comment in comment_entries if comment["id"] == comment_id
					),
				)
				selected_post, selected_comment = next(
					(post, comment) for post, comment in comment_entries
					if comment["id"] == selected_comment_id
				)
				with st.form("edit_comment_form"):
					edited_comment_author = st.text_input(
						"댓글 작성자 수정", value=selected_comment.get("author", ""), max_chars=20
					)
					edited_comment_content = st.text_area(
						"댓글 내용 수정", value=selected_comment.get("content", ""), max_chars=1000
					)
					edit_comment_submitted = st.form_submit_button("댓글 수정 저장", type="primary", use_container_width=True)
				if edit_comment_submitted:
					if not edited_comment_author.strip() or not edited_comment_content.strip():
						st.warning("댓글 작성자와 내용을 비워 둘 수 없습니다.")
					else:
						selected_comment.update({
							"author": edited_comment_author.strip(),
							"content": edited_comment_content.strip(),
						})
						save_posts(posts)
						st.success("댓글을 수정했습니다.")
						st.rerun()
				with st.form("delete_comment_form"):
					confirm_comment_delete = st.checkbox("이 댓글을 영구 삭제합니다.")
					delete_comment_submitted = st.form_submit_button("댓글 삭제")
				if delete_comment_submitted:
					if confirm_comment_delete:
						selected_post["comments"] = [
							comment for comment in selected_post["comments"]
							if comment["id"] != selected_comment_id
						]
						save_posts(posts)
						st.success("댓글을 삭제했습니다.")
						st.rerun()
					st.warning("삭제하려면 확인 항목을 선택해 주세요.")
			else:
				st.caption("관리할 댓글이 없습니다.")
