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

import os
import json
import random
import textwrap
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ColorClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    concatenate_videoclips,
)
from moviepy.video.fx import FadeIn, FadeOut, Resize, Crop
from moviepy.audio.fx import MultiplyVolume, AudioFadeIn, AudioFadeOut

load_dotenv()

# ══════════════════════════════════════════════
# 🔧 환경 설정
# ══════════════════════════════════════════════

OUTPUT_DIR = Path("output")
FRAMES_DIR = OUTPUT_DIR / "frames"
BROLL_DIR = OUTPUT_DIR / "broll"
BGM_DIR = Path("assets/bgm")  # BGM 파일들 (사전 준비 필요)

for d in [OUTPUT_DIR, FRAMES_DIR, BROLL_DIR, BGM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

VIDEO_WIDTH, VIDEO_HEIGHT = 1080, 1920

# 한글 폰트 자동 탐색 (OS별)
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",  # Ubuntu nanum
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",  # Ubuntu noto-cjk
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",  # Ubuntu noto-cjk (otf)
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Ubuntu wqy
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",  # macOS
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
    "C:/Windows/Fonts/malgun.ttf",  # Windows
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # fallback (한글 X)
]


def get_font_path() -> str:
    """사용 가능한 한글 폰트 경로 반환"""
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
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.8,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

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
        script = json.loads(result.content)

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

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

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
        data = json.loads(result.content)
        keywords = data["keywords"]

        if len(keywords) != len(sentences):
            # 길이 안 맞으면 패딩
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

    # 시드별로 다른 색상
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
        api_key = os.getenv("PEXELS_API_KEY")

        media_paths = []
        for i, keyword in enumerate(keywords):
            success = False

            if api_key:
                try:
                    headers = {"Authorization": api_key}
                    url = "https://api.pexels.com/videos/search"
                    params = {
                        "query": keyword,
                        "orientation": "portrait",
                        "size": "medium",
                        "per_page": 3,
                    }
                    r = requests.get(url, headers=headers, params=params, timeout=10)

                    if r.status_code == 200:
                        videos = r.json().get("videos", [])
                        if videos:
                            video = random.choice(videos)
                            # HD 화질 우선
                            hd_files = [
                                f
                                for f in video["video_files"]
                                if f.get("quality") == "hd"
                            ]
                            video_file = (
                                hd_files[0] if hd_files else video["video_files"][0]
                            )

                            path = BROLL_DIR / f"broll_{i}.mp4"
                            with requests.get(
                                video_file["link"], stream=True, timeout=30
                            ) as vr:
                                with open(path, "wb") as f:
                                    for chunk in vr.iter_content(8192):
                                        f.write(chunk)

                            media_paths.append({"type": "video", "path": str(path)})
                            print(f"📹 B-roll 다운로드: {keyword}")
                            success = True
                except Exception as e:
                    print(f"⚠️  Pexels 실패 ({keyword}): {e}")

            # Fallback: 그라데이션 이미지
            if not success:
                gradient_path = _create_gradient_background(
                    VIDEO_WIDTH, VIDEO_HEIGHT, hue_seed=i
                )
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
        full_text = " ".join(
            [
                script["hook"],
                *script["body"],
                script["cta"],
            ]
        )

        instruction = script.get("tts_instruction", "활기차고 친근한 톤으로 말해주세요")
        output_path = OUTPUT_DIR / "audio.mp3"

        client = OpenAI()
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="ballad",
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

        words = [
            {"word": w.word, "start": w.start, "end": w.end} for w in transcript.words
        ]

        with open(OUTPUT_DIR / "timestamps.json", "w", encoding="utf-8") as f:
            json.dump(words, f, ensure_ascii=False, indent=2)

        print(f"✅ 타임스탬프 추출: {len(words)}개 단어")
        return json.dumps(words, ensure_ascii=False)

    except Exception as e:
        return f"ERROR: 타임스탬프 추출 실패 - {e}"


# ══════════════════════════════════════════════
# 🎬 영상 합성 헬퍼 함수들
# ══════════════════════════════════════════════


def _prepare_background(media: dict, duration: float):
    """B-roll 미디어를 세로 1080x1920에 맞게 가공"""
    if media["type"] == "video":
        clip = VideoFileClip(media["path"]).with_duration(duration)
    else:  # image
        clip = ImageClip(media["path"]).with_duration(duration)

    # 비율 맞춰 리사이즈 (높이 1920에 맞춤)
    w, h = clip.size
    scale = VIDEO_HEIGHT / h
    new_w = int(w * scale)
    clip = clip.with_effects([Resize((new_w, VIDEO_HEIGHT))])

    # 중앙 크롭 (너무 넓으면 좌우 자르고, 좁으면 검은 배경에 얹기)
    if new_w >= VIDEO_WIDTH:
        x_start = (new_w - VIDEO_WIDTH) // 2
        clip = clip.with_effects(
            [
                Crop(
                    x1=x_start,
                    x2=x_start + VIDEO_WIDTH,
                    y1=0,
                    y2=VIDEO_HEIGHT,
                )
            ]
        )
    else:
        # 가로가 부족하면 1080에 맞춰 늘림
        clip = clip.with_effects([Resize((VIDEO_WIDTH, VIDEO_HEIGHT))])

    return clip


def _build_karaoke_captions(words: list, total_duration: float) -> list:
    """단어들을 5개씩 묶어서 카라오케 자막 클립 생성"""
    if not words:
        return []

    captions = []
    chunk_size = 5

    # 단어들을 5개씩 묶기
    chunks = [words[i : i + chunk_size] for i in range(0, len(words), chunk_size)]

    for chunk in chunks:
        chunk_text = " ".join(w["word"] for w in chunk)
        chunk_start = chunk[0]["start"]
        chunk_end = chunk[-1]["end"]
        chunk_duration = chunk_end - chunk_start
        # 2) 강조 자막 (노란색) - 단어별로 짧게 깜빡
        for w in chunk:
            try:
                highlight = (
                    TextClip(
                        font=FONT_PATH,
                        text=w["word"],
                        font_size=50,  # 더 크게
                        color="yellow",
                        stroke_color="black",
                        stroke_width=6,
                        margin=(30, 40),
                        method="label",
                    )
                    .with_start(w["start"])
                    .with_duration(max(w["end"] - w["start"], 0.1))
                    .with_position(("center", VIDEO_HEIGHT * 0.78))
                )
                captions.append(highlight)
            except Exception as e:
                print(f"⚠️  단어 클립 생성 실패: {w['word']} - {e}")

    return captions


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
        media_list = json.loads(media_json)
        words = json.loads(timestamps_json) if timestamps_json else []

        audio = AudioFileClip(audio_path)
        total_duration = audio.duration

        # 각 B-roll 미디어를 균등 분할
        seg_duration = total_duration / len(media_list)

        # 배경 클립들 만들기
        bg_clips = []
        for i, media in enumerate(media_list):
            bg = _prepare_background(media, seg_duration)
            bg_clips.append(bg)

        background = concatenate_videoclips(bg_clips, method="compose")

        # 어두운 오버레이 (자막 가독성 ↑)
        overlay = (
            ColorClip(
                size=(VIDEO_WIDTH, VIDEO_HEIGHT),
                color=(0, 0, 0),
            )
            .with_opacity(0.35)
            .with_duration(total_duration)
        )

        # 카라오케 자막
        captions = _build_karaoke_captions(words, total_duration)

        # 최종 합성
        layers = [background, overlay] + captions
        final = (
            CompositeVideoClip(layers, size=(VIDEO_WIDTH, VIDEO_HEIGHT))
            .with_duration(total_duration)
            .with_audio(audio)
        )

        output_path = OUTPUT_DIR / output_filename
        final.write_videofile(
            str(output_path),
            fps=30,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=str(OUTPUT_DIR / "temp_audio.m4a"),
            remove_temp=True,
            threads=4,
        )

        # 리소스 정리
        audio.close()
        final.close()
        for c in bg_clips:
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
    BGM 파일은 assets/bgm/{mood}_*.mp3 형태로 사전 준비되어야 합니다.

    Args:
        video_path: assemble_video가 반환한 영상 경로
        mood: "upbeat", "calm", "dramatic", "inspirational" 중 하나
    """
    try:
        # 해당 mood의 BGM 파일 찾기
        bgm_files = list(BGM_DIR.glob(f"{mood}*.mp3"))
        if not bgm_files:
            bgm_files = list(BGM_DIR.glob("*.mp3"))

        if not bgm_files:
            print(
                f"⚠️  BGM 파일이 없어서 BGM 단계를 건너뜁니다. ({BGM_DIR}/{mood}*.mp3 필요)"
            )
            return video_path  # 원본 그대로 반환

        bgm_path = random.choice(bgm_files)
        print(f"🎵 BGM 선택: {bgm_path.name}")

        video = VideoFileClip(video_path)
        bgm = (
            AudioFileClip(str(bgm_path))
            .with_duration(video.duration)
            .with_effects(
                [
                    MultiplyVolume(0.12),  # 음성 안 묻히게 12%
                    AudioFadeIn(0.5),
                    AudioFadeOut(1.0),
                ]
            )
        )

        # 원본 음성 + BGM 믹스
        mixed = CompositeAudioClip([video.audio, bgm])
        final = video.with_audio(mixed)

        output_path = video_path.replace(".mp4", "_bgm.mp4")
        final.write_videofile(
            output_path,
            fps=30,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=str(OUTPUT_DIR / "temp_bgm.m4a"),
            remove_temp=True,
            threads=4,
        )

        video.close()
        bgm.close()
        final.close()

        print(f"✅ BGM 추가 완료: {output_path}")
        return output_path

    except Exception as e:
        return f"ERROR: BGM 추가 실패 - {e}"


@tool
def generate_ai_images(prompts: list[str]) -> str:
    """각 장면에 맞는 이미지를 AI로 생성합니다."""
    client = OpenAI()
    image_paths = []

    for i, prompt in enumerate(prompts):
        response = client.images.generate(
            model="gpt-image-1",  # 최신 모델
            prompt=f"{prompt}, vertical 9:16 composition, cinematic, vibrant",
            size="1024x1536",  # 세로 비율
            quality="high",
            n=1,
        )

        # base64 이미지 저장
        import base64

        img_data = base64.b64decode(response.data[0].b64_json)
        path = OUTPUT_DIR / f"ai_image_{i}.png"
        path.write_bytes(img_data)
        image_paths.append(str(path))

    return json.dumps(image_paths)


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
    generate_ai_images,
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


def build_agent():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
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
- 주제: 코딩 공부의 중요성
- 타겟: 학부모
- 톤: 동기부여, 활기찬
- 출력 파일명: math_video.mp4""")
