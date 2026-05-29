"""
유튜브 쇼츠 자동 생성 LangChain Agent (PRO 버전)
─────────────────────────────────────────────────
✨ 통합 기능:
  🎤 카라오케 자막   - Whisper로 단어별 타임스탬프 → 단어가 음성에 맞춰 강조
  🎬 B-roll 배경     - Pexels API로 주제 관련 영상 자동 다운로드
  🎵 BGM 자동 추가   - 분위기에 맞는 BGM을 음성과 자연스럽게 믹스

📦 스택:
  - LangChain 1.0+ (create_agent)
  - MoviePy 2.0+ (with_* API)
  - OpenAI 최신 SDK (gpt-4o-mini-tts, whisper-1)
"""

import json
import os
import random
import re
import time as _time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from proglog import ProgressBarLogger

from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    concatenate_videoclips,
)
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut, AudioLoop, AudioNormalize, MultiplyVolume
from moviepy.video.fx import Crop, CrossFadeIn, CrossFadeOut, FadeIn, FadeOut, Loop, Resize

load_dotenv()

# ══════════════════════════════════════════════
# 🔧 경로 설정
# ══════════════════════════════════════════════

OUTPUT_DIR = Path("output")
FRAMES_DIR = OUTPUT_DIR / "frames"
BROLL_DIR = OUTPUT_DIR / "broll"
BGM_DIR = Path("assets/bgm")  # BGM 파일들 (사전 준비 필요)

for d in [OUTPUT_DIR, FRAMES_DIR, BROLL_DIR, BGM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════
# 🎬 영상 규격 설정
# ══════════════════════════════════════════════

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
CROSSFADE_DURATION = 0.4   # 클립 전환 크로스페이드 길이(초)
KEN_BURNS_ZOOM_PAD = 1.25  # Ken Burns 줌 여유 배율 (테두리 방지)

# ══════════════════════════════════════════════
# 📝 자막 설정
# ══════════════════════════════════════════════

CAPTION_CHUNK_SIZE = 5      # 카라오케 자막 한 번에 보여줄 단어 수
CAPTION_Y_RATIO = 0.70      # 기본 자막 세로 위치 (화면 비율)
CAPTION_HIGHLIGHT_Y_RATIO = 0.78  # 강조 자막 세로 위치
CAPTION_FONT_SIZE = 72
CAPTION_HIGHLIGHT_FONT_SIZE = 90

# ══════════════════════════════════════════════
# 🎵 오디오 설정
# ══════════════════════════════════════════════

BGM_VOLUME = 0.12           # BGM 볼륨 (TTS 안 묻히게 낮게)
BGM_FADE_IN = 0.5
BGM_FADE_OUT = 1.0
DARK_OVERLAY_OPACITY = 0.35 # 자막 가독성을 위한 어두운 오버레이

# ══════════════════════════════════════════════
# 🖥️ 인코딩 설정 (assemble_video, add_background_music 공통)
# ══════════════════════════════════════════════

ENCODING_PARAMS = dict(
    fps=30,
    codec="libx264",
    audio_codec="aac",
    audio_bitrate="192k",
    bitrate="12M",
    ffmpeg_params=[
        "-preset", "slow",       # 압축 효율 ↑
        "-pix_fmt", "yuv420p",   # 기기 호환성 (필수)
        "-profile:v", "high",
        "-movflags", "+faststart",  # 웹 스트리밍 최적화
    ],
    threads=4,
    remove_temp=True,
)

# ══════════════════════════════════════════════
# 🤖 모델 설정
# ══════════════════════════════════════════════
#
# 역할 분담:
#   Claude Sonnet → 창의적 글쓰기, 한국어 이해, Agent 오케스트레이션
#   Claude Haiku  → 간단한 추출/분류 (빠르고 저렴)
#   GPT           → TTS, Whisper STT (Anthropic에 없는 기능)

CLAUDE_SONNET = "claude-sonnet-4-5"
CLAUDE_HAIKU = "claude-haiku-4-5-20251001"


def _make_claude(model: str, temperature: float = 0.8) -> ChatAnthropic:
    return ChatAnthropic(model=model, temperature=temperature)


def _make_gpt(model: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature)


def _parse_json_safe(raw: str) -> dict:
    """
    Claude는 response_format 미지원 → 가끔 ```json 코드블록으로 감싸서 반환.
    코드블록 유무에 관계없이 안전하게 파싱.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]).strip()
    return json.loads(text)


# ══════════════════════════════════════════════
# 📊 렌더링 진행 콜백 (Streamlit 등 외부 UI 연동)
# ══════════════════════════════════════════════
# 외부에서 set_render_callback()으로 콜백을 등록하면,
# 영상 합성 중 프레임 진행률(0.0~1.0)을 실시간으로 받을 수 있음.

_render_callback = None  # callable(stage:str, frac:float, eta:float) | None


def set_render_callback(cb):
    """렌더링 진행 콜백 등록. cb(stage, frac, eta_seconds)"""
    global _render_callback
    _render_callback = cb


class RenderProgressLogger(ProgressBarLogger):
    """
    MoviePy write_videofile의 진행률을 가로채 외부 콜백으로 전달.

    MoviePy는 'chunk'(오디오)와 'frame_index'(비디오) 두 bar를 사용하는데,
    오디오는 순식간이고 비디오 프레임 렌더링이 대부분의 시간을 차지함.
    따라서 'frame_index'만 추적해 진행률이 0→100%로 매끄럽게 흐르게 함.
    """

    def __init__(self, stage: str = "렌더링"):
        super().__init__()
        self.stage = stage
        self._start = None

    def bars_callback(self, bar, attr, value, old_value=None):
        if attr != "index" or bar != "frame_index" or _render_callback is None:
            return
        info = self.bars.get(bar, {})
        total = info.get("total", 0)
        if not total:
            return
        if self._start is None:
            self._start = _time.time()
        elapsed = _time.time() - self._start
        frac = min(value / total, 1.0)
        eta = (elapsed / frac * (1 - frac)) if frac > 0.02 else 0.0
        try:
            _render_callback(self.stage, frac, eta)
        except Exception:
            pass  # UI 콜백 실패가 렌더링을 막지 않도록


def _make_logger(stage: str):
    """콜백이 등록돼 있으면 진행 로거를, 아니면 None(기본 동작) 반환"""
    return RenderProgressLogger(stage) if _render_callback else None


# ══════════════════════════════════════════════
# 🔤 폰트 설정
# ══════════════════════════════════════════════

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",  # Ubuntu nanum
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",  # Ubuntu noto-cjk
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",  # Ubuntu noto-cjk (otf)
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",         # Ubuntu wqy
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",   # macOS
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",           # macOS
    "C:/Windows/Fonts/malgun.ttf",                          # Windows
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", # fallback (한글 X)
]


def get_font_path() -> str:
    for fp in FONT_CANDIDATES:
        if Path(fp).exists():
            return fp
    raise FileNotFoundError("사용 가능한 폰트를 찾을 수 없습니다")


FONT_PATH = get_font_path()


# ══════════════════════════════════════════════
# ✅ Tool 1: 스크립트 생성
# ══════════════════════════════════════════════


@tool
def generate_script(topic: str, target: str, tone: str) -> str:
    """
    유튜브 쇼츠용 스크립트를 JSON으로 생성합니다.

    Args:
        topic: 영상 주제
        target: 타겟 시청자
        tone: 영상 톤
    """
    llm = _make_claude(CLAUDE_SONNET, temperature=0.8)

    prompt = f"""유튜브 쇼츠 스크립트를 JSON으로 작성하세요.

주제: {topic}
타겟: {target}
톤: {tone}

규칙:
- 40~50초 분량
- 첫 3초 안에 시청자 사로잡기
- 본문 3~5개 짧은 문장
- mood는 "upbeat", "calm", "dramatic", "inspirational" 중 선택

JSON 형식:
{{
  "title": "제목 (60자 이하)",
  "hook": "첫 3초 훅 문장",
  "body": ["내용1", "내용2", "내용3"],
  "cta": "마무리 CTA",
  "hashtags": ["#태그1", "#태그2", "#태그3"],
  "mood": "upbeat",
  "tts_instruction": "음성 톤 지시 (예: 활기차고 빠르게)"
}}"""

    try:
        result = llm.invoke(prompt)
        script = _parse_json_safe(result.content)

        required = {"title", "hook", "body", "cta", "hashtags", "mood"}
        if not required.issubset(script.keys()):
            return f"ERROR: 필수 키 누락 - {required - script.keys()}"

        with open(OUTPUT_DIR / "script.json", "w", encoding="utf-8") as f:
            json.dump(script, f, ensure_ascii=False, indent=2)

        print(f"✅ 스크립트 생성: {script['title']}")
        return json.dumps(script, ensure_ascii=False)

    except json.JSONDecodeError as e:
        return f"ERROR: JSON 파싱 실패 - {e}"
    except Exception as e:
        return f"ERROR: 스크립트 생성 실패 - {e}"


# ══════════════════════════════════════════════
# ✅ Tool 2: 시각화 키워드 추출
# ══════════════════════════════════════════════


@tool
def extract_visual_keywords(script_json: str) -> str:
    """
    스크립트의 각 문장에서 영상 검색에 쓸 시각적 키워드를 추출합니다 (영어).

    Args:
        script_json: generate_script가 반환한 JSON 문자열
    """
    try:
        script = json.loads(script_json)
        sentences = [script["hook"]] + script["body"] + [script["cta"]]

        llm = _make_claude(CLAUDE_HAIKU, temperature=0.3)

        prompt = f"""각 문장에 어울리는 영상 검색 키워드를 영어로 추출하세요.
Pexels에서 검색할 거니까 간단하고 일반적인 단어로!

문장 리스트:
{json.dumps(sentences, ensure_ascii=False)}

JSON 형식:
{{
  "keywords": ["keyword1", "keyword2", ...]
}}

규칙:
- 문장 개수와 키워드 개수가 정확히 같아야 함
- 각 키워드는 1~3 단어 (예: "laptop coding", "money", "success")
- 너무 추상적이지 않게 (시각화 가능한 단어)"""

        result = llm.invoke(prompt)
        data = _parse_json_safe(result.content)
        keywords = data["keywords"]

        if len(keywords) != len(sentences):
            keywords = (keywords + ["technology"] * len(sentences))[: len(sentences)]

        print(f"✅ 시각 키워드: {keywords}")
        return json.dumps(keywords)

    except Exception as e:
        return f"ERROR: 키워드 추출 실패 - {e}"


# ══════════════════════════════════════════════
# ✅ Tool 3: B-roll 다운로드 (Pexels)
# ══════════════════════════════════════════════


def _create_gradient_background(width: int, height: int, hue_seed: int = 0) -> str:
    """B-roll 못 찾을 때 fallback - 그라데이션 배경 이미지 생성"""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    palettes = [
        [(20, 20, 60), (90, 30, 120)],
        [(15, 40, 70), (60, 110, 150)],
        [(50, 20, 50), (140, 60, 100)],
        [(30, 50, 30), (80, 130, 70)],
    ]
    c1, c2 = palettes[hue_seed % len(palettes)]

    for y in range(height):
        ratio = y / height
        r = int(c1[0] + (c2[0] - c1[0]) * ratio)
        g = int(c1[1] + (c2[1] - c1[1]) * ratio)
        b = int(c1[2] + (c2[2] - c1[2]) * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    path = BROLL_DIR / f"gradient_{hue_seed}.png"
    img.save(path)
    return str(path)


def _download_pexels_video(keyword: str, save_path: Path) -> bool:
    """Pexels에서 키워드로 세로 영상을 검색해 다운로드. 성공 여부 반환."""
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        return False

    try:
        headers = {"Authorization": api_key}
        params = {"query": keyword, "orientation": "portrait", "size": "medium", "per_page": 3}
        r = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params, timeout=10)
        if r.status_code != 200:
            return False

        videos = r.json().get("videos", [])
        if not videos:
            return False

        video = random.choice(videos)
        hd_files = [f for f in video["video_files"] if f.get("quality") == "hd"]
        video_file = hd_files[0] if hd_files else video["video_files"][0]

        with requests.get(video_file["link"], stream=True, timeout=30) as vr:
            with open(save_path, "wb") as f:
                for chunk in vr.iter_content(8192):
                    f.write(chunk)
        return True

    except Exception as e:
        print(f"⚠️  Pexels 실패 ({keyword}): {e}")
        return False


@tool
def fetch_broll_videos(keywords_json: str) -> str:
    """
    Pexels API로 키워드별 세로 영상을 다운로드합니다.
    API 키가 없거나 영상을 못 찾으면 그라데이션 배경 이미지로 대체합니다.

    Args:
        keywords_json: extract_visual_keywords가 반환한 키워드 JSON 배열
    """
    try:
        keywords = json.loads(keywords_json)
        media_paths = []

        for i, keyword in enumerate(keywords):
            video_path = BROLL_DIR / f"broll_{i}.mp4"
            if _download_pexels_video(keyword, video_path):
                media_paths.append({"type": "video", "path": str(video_path)})
                print(f"📹 B-roll 다운로드: {keyword}")
            else:
                gradient_path = _create_gradient_background(VIDEO_WIDTH, VIDEO_HEIGHT, hue_seed=i)
                media_paths.append({"type": "image", "path": gradient_path})
                print(f"🎨 그라데이션 배경 생성: {keyword}")

        return json.dumps(media_paths)

    except Exception as e:
        return f"ERROR: B-roll 다운로드 실패 - {e}"


# ══════════════════════════════════════════════
# ✅ Tool 4: TTS 음성 생성
# ══════════════════════════════════════════════


@tool
def generate_tts(script_json: str) -> str:
    """
    스크립트에서 TTS 음성 파일을 생성합니다.

    Args:
        script_json: generate_script가 반환한 JSON 문자열
    """
    try:
        script = json.loads(script_json)
        full_text = " ".join([script["hook"], *script["body"], script["cta"]])
        instruction = script.get("tts_instruction", "활기차고 친근한 톤으로 말해주세요")
        output_path = OUTPUT_DIR / "audio.mp3"

        client = OpenAI()
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="nova",
            input=full_text,
            instructions=instruction,
        ) as response:
            response.stream_to_file(output_path)

        print(f"✅ TTS 생성: {output_path}")
        return str(output_path)

    except Exception as e:
        return f"ERROR: TTS 생성 실패 - {e}"


# ══════════════════════════════════════════════
# ✅ Tool 5: 단어별 타임스탬프 추출 (Whisper)
# ══════════════════════════════════════════════


@tool
def get_word_timestamps(audio_path: str) -> str:
    """
    Whisper로 음성에서 단어별 타임스탬프를 추출합니다.
    카라오케 스타일 자막 생성에 사용됩니다.

    Args:
        audio_path: TTS 음성 파일 경로
    """
    try:
        client = OpenAI()
        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["word"],
                language="ko",
            )

        words = [{"word": w.word, "start": w.start, "end": w.end} for w in transcript.words]

        with open(OUTPUT_DIR / "timestamps.json", "w", encoding="utf-8") as f:
            json.dump(words, f, ensure_ascii=False, indent=2)

        print(f"✅ 타임스탬프 추출: {len(words)}개 단어")
        return json.dumps(words, ensure_ascii=False)

    except Exception as e:
        return f"ERROR: 타임스탬프 추출 실패 - {e}"


# ══════════════════════════════════════════════
# 🎬 영상 합성 헬퍼 함수들
# ══════════════════════════════════════════════


def _ken_burns(clip, duration: float, index: int = 0):
    """
    Ken Burns 효과: index 짝수=줌인(1.0→1.15), 홀수=줌아웃(1.15→1.0)으로 번갈아 적용.
    """
    if index % 2 == 0:
        return clip.resized(lambda t: 1.0 + 0.15 * (t / duration))
    else:
        return clip.resized(lambda t: 1.15 - 0.15 * (t / duration))


def _prepare_background(media: dict, duration: float, index: int = 0):
    """B-roll 미디어를 세로 1080x1920에 맞게 가공 + Ken Burns 효과"""
    if media["type"] == "video":
        clip = VideoFileClip(media["path"]).without_audio()
        if clip.duration < duration:
            clip = clip.with_effects([Loop(duration=duration)])
        else:
            clip = clip.with_duration(duration)
    else:
        clip = ImageClip(media["path"]).with_duration(duration)

    # 높이 1920 기준으로 리사이즈
    w, h = clip.size
    scale = VIDEO_HEIGHT / h
    new_w = int(w * scale)
    clip = clip.with_effects([Resize((new_w, VIDEO_HEIGHT))])

    # 중앙 크롭 (너무 넓으면 좌우 자르고, 좁으면 늘리기)
    if new_w >= VIDEO_WIDTH:
        x_start = (new_w - VIDEO_WIDTH) // 2
        clip = clip.with_effects([Crop(x1=x_start, x2=x_start + VIDEO_WIDTH, y1=0, y2=VIDEO_HEIGHT)])
    else:
        clip = clip.with_effects([Resize((VIDEO_WIDTH, VIDEO_HEIGHT))])

    # Ken Burns: 줌 여유 공간 확보 후 적용, 다시 정확히 크롭
    pad_w = int(VIDEO_WIDTH * KEN_BURNS_ZOOM_PAD)
    pad_h = int(VIDEO_HEIGHT * KEN_BURNS_ZOOM_PAD)
    clip = clip.with_effects([Resize((pad_w, pad_h))])
    clip = _ken_burns(clip, duration, index)
    clip = clip.with_effects([Crop(width=VIDEO_WIDTH, height=VIDEO_HEIGHT, x_center=pad_w // 2, y_center=pad_h // 2)])

    return clip


def _pop_scale(t):
    """단어가 톡 튀어오르는 효과 (작게→크게→안정)"""
    if t < 0.12:
        return 0.6 + (1.25 - 0.6) * (t / 0.12)
    elif t < 0.20:
        return 1.25 - 0.25 * ((t - 0.12) / 0.08)
    return 1.0


def _build_karaoke_captions(words: list, total_duration: float) -> list:
    """단어들을 CAPTION_CHUNK_SIZE개씩 묶어서 카라오케 자막 클립 생성"""
    if not words:
        return []

    captions = []
    chunks = [words[i: i + CAPTION_CHUNK_SIZE] for i in range(0, len(words), CAPTION_CHUNK_SIZE)]

    for chunk in chunks:
        chunk_text = " ".join(w["word"] for w in chunk)
        chunk_start = chunk[0]["start"]
        chunk_end = chunk[-1]["end"]
        chunk_duration = chunk_end - chunk_start

        # 기본 자막 (흰색) - 청크 전체 시간 표시
        # margin=(0, 30): 한글 받침 + stroke 잘림 방지
        base = (
            TextClip(
                font=FONT_PATH,
                text=chunk_text,
                font_size=CAPTION_FONT_SIZE,
                color="white",
                stroke_color="black",
                stroke_width=5,
                method="caption",
                size=(900, None),
                text_align="center",
                margin=(0, 30),
            )
            .with_start(chunk_start)
            .with_duration(chunk_duration)
            .with_position(("center", VIDEO_HEIGHT * CAPTION_Y_RATIO))
        )
        captions.append(base)

        # 강조 자막 (노란색) - 단어별 Pop 애니메이션
        # margin=(30, 40): 큰 폰트 + 두꺼운 stroke 여백
        for w in chunk:
            try:
                highlight = (
                    TextClip(
                        font=FONT_PATH,
                        text=w["word"],
                        font_size=CAPTION_HIGHLIGHT_FONT_SIZE,
                        color="yellow",
                        stroke_color="black",
                        stroke_width=6,
                        method="label",
                        margin=(30, 40),
                    )
                    .with_start(w["start"])
                    .with_duration(max(w["end"] - w["start"], 0.1))
                    .resized(_pop_scale)
                    .with_position(("center", VIDEO_HEIGHT * CAPTION_HIGHLIGHT_Y_RATIO))
                )
                captions.append(highlight)
            except Exception as e:
                print(f"⚠️  단어 클립 생성 실패: {w['word']} - {e}")

    return captions


def _build_background_layer(media_list: list, total_duration: float):
    """B-roll 미디어 리스트로 크로스페이드 배경 클립 생성"""
    seg_duration = total_duration / len(media_list)
    bg_clips = []

    for i, media in enumerate(media_list):
        # 크로스페이드 겹침을 위해 segment를 약간 늘림
        seg = seg_duration + (CROSSFADE_DURATION if i > 0 else 0)
        bg = _prepare_background(media, seg, index=i)
        if i > 0:
            bg = bg.with_effects([CrossFadeIn(CROSSFADE_DURATION)])
        bg_clips.append(bg)

    background = concatenate_videoclips(bg_clips, method="compose", padding=-CROSSFADE_DURATION)
    background = background.without_audio()
    return background.with_duration(total_duration)


# ══════════════════════════════════════════════
# ✅ Tool 6: B-roll + 카라오케 자막 합성
# ══════════════════════════════════════════════


@tool
def assemble_video(
    media_json: str,
    audio_path: str,
    timestamps_json: str,
    output_filename: str = "shorts_pro.mp4",
) -> str:
    """
    B-roll 배경 + 카라오케 자막 + 음성을 합쳐 영상을 만듭니다.

    Args:
        media_json: fetch_broll_videos가 반환한 미디어 JSON
        audio_path: generate_tts가 반환한 음성 경로
        timestamps_json: get_word_timestamps가 반환한 단어 타임스탬프
        output_filename: 출력 파일명
    """
    try:
        for name, val in [("media_json", media_json), ("audio_path", audio_path)]:
            if val.startswith("ERROR:"):
                return f"ERROR: {name}에 이전 Tool 오류가 전달됨 - {val}"

        if not Path(audio_path).exists():
            return f"ERROR: 음성 파일 없음 - {audio_path}"

        media_list = json.loads(media_json)
        words = (
            json.loads(timestamps_json)
            if timestamps_json and not timestamps_json.startswith("ERROR:")
            else []
        )
        if not media_list:
            return "ERROR: 미디어 리스트가 비어있습니다"

        audio = AudioFileClip(audio_path)
        total_duration = audio.duration
        print(f"📊 TTS 음성 길이: {total_duration:.2f}초")

        background = _build_background_layer(media_list, total_duration)

        overlay = (
            ColorClip(size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=(0, 0, 0))
            .with_opacity(DARK_OVERLAY_OPACITY)
            .with_duration(total_duration)
        )

        captions = _build_karaoke_captions(words, total_duration)

        final = CompositeVideoClip(
            [background, overlay] + captions, size=(VIDEO_WIDTH, VIDEO_HEIGHT)
        ).with_duration(total_duration)
        final = final.with_audio(audio)

        if final.audio is None:
            return "ERROR: 합성 후 audio가 None입니다"

        output_path = OUTPUT_DIR / output_filename
        final.write_videofile(
            str(output_path),
            **ENCODING_PARAMS,
            temp_audiofile=str(OUTPUT_DIR / "temp_audio.m4a"),
            logger=_make_logger("영상 합성"),
        )

        audio.close()
        final.close()
        for c in background.clips if hasattr(background, "clips") else []:
            c.close()

        print(f"✅ 영상 합성 완료: {output_path}")
        return str(output_path)

    except Exception as e:
        return f"ERROR: 영상 합성 실패 - {e}"


# ══════════════════════════════════════════════
# ✅ Tool 7: BGM 추가
# ══════════════════════════════════════════════


@tool
def add_background_music(video_path: str, mood: str = "upbeat") -> str:
    """
    영상에 분위기에 맞는 BGM을 추가합니다.
    BGM이 영상보다 짧으면 자동으로 루프 처리됩니다.

    Args:
        video_path: assemble_video가 반환한 영상 경로
        mood: "upbeat", "calm", "dramatic", "inspirational" 중 하나
    """
    try:
        if video_path.startswith("ERROR:"):
            return f"ERROR: 이전 단계 오류 - {video_path}"
        if not Path(video_path).exists():
            return f"ERROR: 영상 파일 없음 - {video_path}"

        bgm_files = list(BGM_DIR.glob(f"{mood}*.mp3")) or list(BGM_DIR.glob("*.mp3"))
        if not bgm_files:
            print("⚠️  BGM 파일이 없어서 BGM 단계를 건너뜁니다.")
            return video_path

        bgm_path = random.choice(bgm_files)
        print(f"🎵 BGM 선택: {bgm_path.name}")

        video = VideoFileClip(video_path)
        print(f"📊 영상 길이: {video.duration:.2f}초")

        if video.audio is None:
            video.close()
            return "ERROR: 입력 영상에 오디오 트랙이 없습니다 (assemble_video 단계 문제)"

        bgm_raw = AudioFileClip(str(bgm_path))
        print(f"📊 BGM 원본 길이: {bgm_raw.duration:.2f}초")

        if bgm_raw.duration < video.duration:
            print("🔁 BGM이 짧음 → 루프로 늘림")
            bgm = bgm_raw.with_effects([AudioLoop(duration=video.duration)])
        else:
            bgm = bgm_raw.with_duration(video.duration)

        bgm = bgm.with_effects([MultiplyVolume(BGM_VOLUME), AudioFadeIn(BGM_FADE_IN), AudioFadeOut(BGM_FADE_OUT)])

        tts_audio = video.audio.with_effects([AudioNormalize()])
        mixed = CompositeAudioClip([tts_audio, bgm])
        final = video.with_audio(mixed)

        output_path = video_path.replace(".mp4", "_bgm.mp4")
        final.write_videofile(
            output_path,
            **ENCODING_PARAMS,
            temp_audiofile=str(OUTPUT_DIR / "temp_bgm.m4a"),
            logger=_make_logger("배경음악 입히기"),
        )

        video.close()
        bgm_raw.close()
        final.close()

        print(f"✅ BGM 추가 완료: {output_path}")
        return output_path

    except Exception as e:
        return f"ERROR: BGM 추가 실패 - {e}"


# ══════════════════════════════════════════════
# 🤖 Agent 구성
# ══════════════════════════════════════════════

tools = [
    generate_script,
    extract_visual_keywords,
    fetch_broll_videos,
    generate_tts,
    get_word_timestamps,
    assemble_video,
    add_background_music,
]

SYSTEM_PROMPT = """당신은 유튜브 쇼츠 영상 자동 제작 전문 AI입니다.

다음 순서대로 도구들을 정확히 호출해서 쇼츠를 완성하세요:

1️⃣  generate_script(topic, target, tone)
    → 스크립트 JSON 반환 (mood 포함)

2️⃣  extract_visual_keywords(script_json)
    → 시각 키워드 배열 반환

3️⃣  fetch_broll_videos(keywords_json)
    → B-roll 미디어 JSON 반환

4️⃣  generate_tts(script_json)
    → 음성 파일 경로 반환

5️⃣  get_word_timestamps(audio_path)
    → 단어별 타임스탬프 JSON 반환

6️⃣  assemble_video(media_json, audio_path, timestamps_json, output_filename)
    → 합성된 영상 경로 반환

7️⃣  add_background_music(video_path, mood)
    → 1번 스크립트의 mood 값을 사용, 최종 영상 경로 반환

규칙:
- Tool이 'ERROR:'로 시작하는 값을 반환하면 즉시 멈추고 사용자에게 보고하세요.
- 사용자가 출력 파일명을 지정하면 그대로 사용하세요.
- 마지막에 최종 파일 경로와 영상 정보(제목, 해시태그 등)를 요약해주세요."""


def extract_params_from_request(user_request: str) -> dict:
    """
    사용자의 자연어 요청에서 영상 제작 파라미터를 추출합니다.

    Args:
        user_request: 사용자가 자연어로 입력한 요청

    Returns:
        {"topic": ..., "target": ..., "tone": ..., "output_filename": ...}
    """
    llm = _make_claude(CLAUDE_SONNET, temperature=0.4)

    prompt = f"""사용자의 요청에서 유튜브 쇼츠 제작에 필요한 정보를 추출하세요.
명시되지 않은 항목은 요청 내용에 가장 어울리도록 합리적으로 추론해서 채우세요.

사용자 요청: "{user_request}"

JSON 형식으로만 응답:
{{
  "topic": "영상 주제 (구체적으로)",
  "target": "타겟 시청자 (예: 20대 직장인, 코딩 입문자)",
  "tone": "영상 톤 (예: 동기부여, 친근함, 유머러스)",
  "output_filename": "영문 소문자+언더스코어 파일명.mp4 (예: python_tips.mp4)"
}}

규칙:
- output_filename은 반드시 영문 소문자, 숫자, 언더스코어만 사용하고 .mp4로 끝나야 함
- 모든 값은 비어있으면 안 됨"""

    defaults = {
        "topic": user_request[:50],
        "target": "일반 시청자",
        "tone": "친근하고 활기차게",
        "output_filename": "shorts_output.mp4",
    }

    try:
        result = llm.invoke(prompt)
        params = _parse_json_safe(result.content)

        for key, default in defaults.items():
            if not params.get(key):
                params[key] = default

        fname = params["output_filename"]
        if not fname.endswith(".mp4"):
            fname += ".mp4"
        stem = re.sub(r"[^a-zA-Z0-9_]", "_", fname[:-4]).strip("_") or "shorts_output"
        params["output_filename"] = f"{stem}.mp4"
        return params

    except Exception as e:
        return {**defaults, "_error": str(e)}


def build_agent():
    model = _make_claude(CLAUDE_SONNET, temperature=0)
    return create_agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT)


# ══════════════════════════════════════════════
# 🚀 실행
# ══════════════════════════════════════════════


def run(user_request: str):
    agent = build_agent()
    print(f"\n{'=' * 60}")
    print(f"🎬 쇼츠 PRO 제작 요청")
    print(f"{'=' * 60}\n{user_request}\n")

    result = agent.invoke({"messages": [{"role": "user", "content": user_request}]})

    print("\n" + "=" * 60)
    print("🎉 최종 결과")
    print("=" * 60)
    print(result["messages"][-1].content)
    return result


if __name__ == "__main__":
    run("""다음 조건으로 유튜브 쇼츠를 만들어주세요:
- 주제: 파이썬으로 돈 버는 5가지 방법
- 타겟: 취준생, 직장인
- 톤: 동기부여, 실용적
- 출력 파일명: python_money_pro.mp4""")
