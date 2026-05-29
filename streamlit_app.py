"""
🎬 유튜브 쇼츠 자동 제작 - Streamlit 프론트엔드
─────────────────────────────────────────────────
사용법:
    streamlit run streamlit_app.py

기능:
  - 자연어 입력 → 주제/타겟/톤/파일명 자동 추출
  - 단계별 실시간 진행 상황 표시
  - 완성된 영상 미리보기 + 다운로드
"""

import json
import time
from pathlib import Path

import streamlit as st

# 기존 파이프라인 모듈 import (Tool들 + 헬퍼)
import shorts_agent_pro as sap


# ══════════════════════════════════════════════
# 페이지 설정
# ══════════════════════════════════════════════

st.set_page_config(
    page_title="쇼츠 자동 제작기",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 유튜브 쇼츠 자동 제작기")
st.caption("자연어로 원하는 영상을 설명하면 AI가 알아서 만들어드려요!")


# ══════════════════════════════════════════════
# 파이프라인 단계 정의
# ══════════════════════════════════════════════
# (라벨, 진행 메시지, 예상 소요 가중치)
# 가중치는 상대적 비율 — 영상 합성이 가장 무거움
PIPELINE_STEPS = [
    ("요청 분석", "🧠 무슨 영상을 원하시는지 이해하는 중...", 3),
    ("스크립트 작성", "✍️ 시청자를 사로잡을 대본을 쓰는 중...", 6),
    ("키워드 추출", "🔍 어울리는 영상 장면을 고르는 중...", 3),
    ("배경 영상 수집", "🎬 멋진 배경 클립을 모으는 중...", 8),
    ("AI 성우 녹음", "🎤 또박또박 음성을 녹음하는 중...", 6),
    ("자막 타이밍 분석", "⏱️ 단어 하나하나 타이밍을 맞추는 중...", 5),
    ("영상 합성", "🎞️ 배경·자막·음성을 하나로 합치는 중...", 45),
    ("배경음악 입히기", "🎵 분위기를 살릴 BGM을 더하는 중...", 18),
]
# 가중치 합계 (ETA 계산용)
TOTAL_WEIGHT = sum(s[2] for s in PIPELINE_STEPS)


def format_eta(seconds: float) -> str:
    """초를 사람이 읽기 좋은 형태로 변환"""
    if seconds <= 0:
        return "거의 다 됐어요"
    if seconds < 60:
        return f"약 {int(seconds)}초 남음"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"약 {mins}분 {secs}초 남음"


def run_pipeline(user_request: str, status_box, progress_bar, eta_box):
    """
    전체 파이프라인을 단계별로 실행하며 진행 상황 + ETA를 UI에 업데이트.
    각 단계의 결과가 'ERROR:'로 시작하면 즉시 중단하고 예외 발생.

    Returns:
        (최종 영상 경로, 스크립트 dict)
    """
    total_steps = len(PIPELINE_STEPS)
    pipeline_start = time.time()

    # 완료된 단계들의 누적 가중치 → 전체 진행률 계산
    done_weight = [0]  # 리스트로 감싸 클로저에서 수정 가능하게

    def weight_before(step_idx: int) -> int:
        return sum(PIPELINE_STEPS[i][2] for i in range(step_idx))

    def overall_eta(step_idx: int, sub_frac: float = 0.0) -> float:
        """전체 남은 예상 시간(초). 경과시간 기반으로 보정."""
        completed_w = weight_before(step_idx) + PIPELINE_STEPS[step_idx][2] * sub_frac
        frac = completed_w / TOTAL_WEIGHT
        elapsed = time.time() - pipeline_start
        if frac > 0.03:
            return elapsed / frac * (1 - frac)
        # 초반엔 경험적 추정 (가중치 합을 초로 환산)
        return TOTAL_WEIGHT * (1 - frac)

    def update(step_idx: int, sub_frac: float = 0.0):
        label, msg = PIPELINE_STEPS[step_idx][0], PIPELINE_STEPS[step_idx][1]
        overall_frac = (
            weight_before(step_idx) + PIPELINE_STEPS[step_idx][2] * sub_frac
        ) / TOTAL_WEIGHT
        progress_bar.progress(
            min(overall_frac, 0.99),
            text=f"{label} ({step_idx + 1}/{total_steps})",
        )
        status_box.write(msg)
        eta_box.info(f"⏳ 완성까지 {format_eta(overall_eta(step_idx, sub_frac))}")

    def check_error(result, step_name: str):
        if isinstance(result, str) and result.startswith("ERROR:"):
            raise RuntimeError(f"[{step_name}] {result}")
        return result

    # ⭐ 렌더링(영상 합성/BGM) 단계의 정밀 진행률을 받는 콜백 등록
    # MoviePy 프레임 진행률(frac)과 그 단계의 ETA를 받아 UI 갱신
    def render_cb(stage: str, frac: float, eta: float):
        # 어느 단계인지에 따라 step_idx 매핑
        step_idx = 6 if stage == "영상 합성" else 7
        label = PIPELINE_STEPS[step_idx][0]
        overall_frac = (
            weight_before(step_idx) + PIPELINE_STEPS[step_idx][2] * frac
        ) / TOTAL_WEIGHT
        progress_bar.progress(
            min(overall_frac, 0.99),
            text=f"{label} {int(frac * 100)}% ({step_idx + 1}/{total_steps})",
        )
        eta_box.info(f"⏳ 완성까지 {format_eta(overall_eta(step_idx, frac))}")

    sap.set_render_callback(render_cb)

    try:
        # ── 0. 요청 분석 (자연어 → 파라미터) ──────────────
        update(0)
        params = sap.extract_params_from_request(user_request)
        status_box.write(
            f"📋 **추출된 설정**\n"
            f"- 주제: {params['topic']}\n"
            f"- 타겟: {params['target']}\n"
            f"- 톤: {params['tone']}\n"
            f"- 파일명: `{params['output_filename']}`"
        )

        # ── 1. 스크립트 작성 ─────────────────────────────
        update(1)
        script_json = check_error(
            sap.generate_script.invoke(
                {
                    "topic": params["topic"],
                    "target": params["target"],
                    "tone": params["tone"],
                }
            ),
            "스크립트 작성",
        )
        script = json.loads(script_json)
        status_box.write(f"📝 제목: **{script['title']}**")

        # ── 2. 키워드 추출 ───────────────────────────────
        update(2)
        keywords_json = check_error(
            sap.extract_visual_keywords.invoke({"script_json": script_json}),
            "키워드 추출",
        )

        # ── 3. 배경 영상 수집 ────────────────────────────
        update(3)
        media_json = check_error(
            sap.fetch_broll_videos.invoke({"keywords_json": keywords_json}),
            "배경 영상 수집",
        )

        # ── 4. AI 성우 녹음 (TTS) ───────────────────────
        update(4)
        audio_path = check_error(
            sap.generate_tts.invoke({"script_json": script_json}),
            "AI 성우 녹음",
        )

        # ── 5. 자막 타이밍 분석 (Whisper) ───────────────
        update(5)
        timestamps_json = check_error(
            sap.get_word_timestamps.invoke({"audio_path": audio_path}),
            "자막 타이밍 분석",
        )

        # ── 6. 영상 합성 (정밀 ETA: render_cb가 실시간 갱신) ─
        update(6)
        video_path = check_error(
            sap.assemble_video.invoke(
                {
                    "media_json": media_json,
                    "audio_path": audio_path,
                    "timestamps_json": timestamps_json,
                    "output_filename": params["output_filename"],
                }
            ),
            "영상 합성",
        )

        # ── 7. 배경음악 입히기 ──────────────────────────
        update(7)
        mood = script.get("mood", "upbeat")
        final_path = check_error(
            sap.add_background_music.invoke(
                {
                    "video_path": video_path,
                    "mood": mood,
                }
            ),
            "배경음악 입히기",
        )

        progress_bar.progress(1.0, text=f"완료! ({total_steps}/{total_steps})")
        eta_box.success("✅ 완성됐어요!")
        return final_path, script

    finally:
        # 콜백 해제 (다음 실행에 영향 없도록)
        sap.set_render_callback(None)


# ══════════════════════════════════════════════
# 메인 UI
# ══════════════════════════════════════════════

# 세션 상태 초기화
if "result_path" not in st.session_state:
    st.session_state.result_path = None
    st.session_state.result_script = None

# 입력 영역
user_request = st.text_area(
    "어떤 영상을 만들까요?",
    placeholder="예) 코딩 입문자를 위해 파이썬으로 돈 버는 5가지 방법을 동기부여되게 알려주는 영상 만들어줘",
    height=100,
)

example_cols = st.columns(3)
examples = [
    "ESP32로 만드는 신기한 IoT 프로젝트 3가지를 흥미진진하게 소개하는 영상",
    "초등학생도 이해하는 아두이노 기초를 친근하게 설명하는 영상",
    "개발자가 알아야 할 생산성 꿀팁을 빠르게 정리한 영상",
]
for col, ex in zip(example_cols, examples):
    if col.button(ex[:18] + "...", use_container_width=True):
        user_request = ex
        st.session_state.pending_request = ex

# 버튼으로 예시를 눌렀을 때 처리
if "pending_request" in st.session_state and not user_request:
    user_request = st.session_state.pending_request

generate = st.button("🚀 영상 만들기", type="primary", use_container_width=True)

# ══════════════════════════════════════════════
# 실행
# ══════════════════════════════════════════════

if generate:
    if not user_request or not user_request.strip():
        st.warning("먼저 어떤 영상을 원하는지 입력해주세요! 😊")
        st.stop()

    progress_bar = st.progress(0.0, text="준비 중...")
    eta_box = st.empty()  # ETA 표시 영역

    with st.status("🎬 영상 제작 중...", expanded=True) as status:
        start_time = time.time()
        try:
            final_path, script = run_pipeline(
                user_request, status, progress_bar, eta_box
            )
            elapsed = time.time() - start_time

            st.session_state.result_path = final_path
            st.session_state.result_script = script

            status.update(
                label=f"✅ 완성! ({elapsed:.0f}초 소요)",
                state="complete",
                expanded=False,
            )
        except Exception as e:
            status.update(label="❌ 제작 중 오류 발생", state="error")
            st.error(f"오류가 발생했어요: {e}")
            st.info(
                "💡 잠깐 후 다시 시도하거나, 요청을 조금 더 구체적으로 적어보세요. "
                "API 키(.env)와 BGM 파일(assets/bgm/)이 설정되어 있는지도 확인해주세요."
            )
            st.stop()

# ══════════════════════════════════════════════
# 결과 표시
# ══════════════════════════════════════════════

if st.session_state.result_path and Path(st.session_state.result_path).exists():
    script = st.session_state.result_script
    path = st.session_state.result_path

    st.success("🎉 영상이 완성되었어요!")

    # 제목 + 해시태그 정보
    if script:
        st.subheader(script.get("title", "완성된 쇼츠"))
        tags = script.get("hashtags", [])
        if tags:
            st.write(" ".join(tags))

    # 영상 미리보기
    with open(path, "rb") as f:
        video_bytes = f.read()
    st.video(video_bytes)

    # 다운로드 버튼
    st.download_button(
        label="⬇️ 영상 다운로드",
        data=video_bytes,
        file_name=Path(path).name,
        mime="video/mp4",
        type="primary",
        use_container_width=True,
    )

    # 업로드용 정보 복사 도우미
    if script:
        with st.expander("📋 업로드용 정보 (제목/설명/태그)"):
            st.text_input("제목", script.get("title", ""))
            st.text_area(
                "설명",
                f"{script.get('hook', '')}\n\n{' '.join(script.get('hashtags', []))}",
            )
