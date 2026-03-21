import streamlit as st
import os
import base64
import tempfile
import re
import json
import yt_dlp
import plotly.graph_objects as go
from dotenv import load_dotenv
from groq import Groq
from google import genai
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

load_dotenv()

# ─── Proxy config ─────────────────────────────────────────────────────────────
# Set PROXY_URL in your environment/secrets to route through a residential IP.
# Format: "http://user:pass@host:port" or "http://host:port"
# Free option to test: https://webshare.io (10 free proxies)
# If not set, the app runs without a proxy (will 403 on cloud hosts).
PROXY_URL = None
try:
    PROXY_URL = st.secrets.get("PROXY_URL") or os.getenv("PROXY_URL")
except Exception:
    PROXY_URL = os.getenv("PROXY_URL")

# ─── API Keys ────────────────────────────────────────────────────────────────

def check_api_keys():
    # Streamlit Cloud uses st.secrets; local dev uses .env via os.getenv
    def get_key(name):
        try:
            val = st.secrets[name]   # dict-style access, not callable
            if val:
                return val
        except Exception:
            pass
        return os.getenv(name)

    groq_key   = get_key("GROQ_API_KEY")
    gemini_key = get_key("GEMINI_API_KEY")
    missing = []
    if not groq_key:   missing.append("GROQ_API_KEY")
    if not gemini_key: missing.append("GEMINI_API_KEY")
    return groq_key, gemini_key, missing

groq_key, gemini_key, missing_keys = check_api_keys()
if missing_keys:
    st.error(f"Missing API keys: {', '.join(missing_keys)}")
    st.stop()

client_groq   = Groq(api_key=groq_key)
client_gemini = genai.Client(api_key=gemini_key)

# ─── Brand channel detection ─────────────────────────────────────────────────

# Known official brand keywords — channels containing these are skipped
BRAND_KEYWORDS = [
    "samsung", "apple", "google", "sony", "huawei", "xiaomi", "oppo", "oneplus",
    "motorola", "nokia", "lg electronics", "microsoft", "official", "officiel",
    "channel", "records", "music", "corp", "inc", "ltd", "gmbh", "s.a.",
]

def is_brand_channel(channel: str, product: str) -> bool:
    """Return True if the channel looks like an official brand or manufacturer."""
    if not channel:
        return False
    ch = channel.lower().strip()

    # Extract the first word(s) of the product name as the likely brand
    product_brand = product.lower().split()[0] if product else ""

    # If channel name IS the brand name or starts with it → skip
    if product_brand and (ch == product_brand or ch.startswith(product_brand)):
        return True

    # Generic official-sounding channel keywords
    for kw in BRAND_KEYWORDS:
        if kw in ch:
            return True

    return False

# ─── Video search ─────────────────────────────────────────────────────────────

def search_videos(query: str, platform: str, max_results: int = 5) -> list[dict]:
    """
    platform="youtube" → long-form independent reviews (1–15 min)
    platform="shorts"  → true YouTube Shorts (≤60 s, /shorts/ URL format)
    Both exclude official brand channels.
    """
    if platform == "youtube":
        # "review" + "-official" biases toward independent reviewers
        search_url   = f"ytsearch{max_results * 4}:{query} review -official"
        max_duration = 900
        min_duration = 60
    else:
        # "#shorts" is the canonical tag creators use for Shorts
        search_url   = f"ytsearch{max_results * 4}:{query} #shorts review"
        max_duration = 60   # true Shorts are ≤ 60 s
        min_duration = 10

    ydl_opts = {"quiet": True, "extract_flat": True, "no_warnings": True}
    if PROXY_URL:
        ydl_opts["proxy"] = PROXY_URL
    results  = []

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info    = ydl.extract_info(search_url, download=False)
            entries = info.get("entries", []) if info else []

            for entry in entries:
                if not entry:
                    continue

                duration = entry.get("duration") or 0
                if duration > max_duration or duration < min_duration:
                    continue

                channel = (entry.get("uploader") or entry.get("channel") or "").strip()

                # Skip official brand channels
                if is_brand_channel(channel, query):
                    continue

                vid_id = entry.get("id", "")
                url    = entry.get("url") or entry.get("webpage_url") or ""
                if not url and vid_id:
                    url = f"https://www.youtube.com/watch?v={vid_id}"

                # For true Shorts, use the /shorts/ URL so it opens in Shorts player
                if platform == "shorts" and vid_id:
                    url = f"https://www.youtube.com/shorts/{vid_id}"

                if not url:
                    continue

                thumbnail = f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg" if vid_id else ""

                results.append({
                    "title":     entry.get("title", "Unknown"),
                    "url":       url,
                    "duration":  duration,
                    "platform":  platform,
                    "thumbnail": thumbnail,
                    "channel":   channel,
                    "views":     entry.get("view_count") or 0,
                })

                if len(results) >= max_results:
                    break

    except Exception as e:
        st.warning(f"Search error ({platform}): {e}")

    return results

# ─── Download + extract audio ─────────────────────────────────────────────────

def download_and_extract_audio(url: str, prefix: str = "tmp") -> tuple:
    # Audio-only download avoids the large video file and is less likely
    # to be blocked. Browser-like headers bypass datacenter IP blocks.
    audio_path = f"{prefix}.mp3"
    ydl_opts = {
        "format":        "bestaudio/best",
        "outtmpl":       audio_path,
        "quiet":         True,
        "overwrites":    True,
        "no_warnings":   True,
        "postprocessors": [{
            "key":            "FFmpegExtractAudio",
            "preferredcodec": "mp3",
        }],
        "http_headers": {
            "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer":         "https://www.youtube.com/",
        },
        "extractor_args": {
            "youtube": {
                "player_client": ["web", "android"],
            }
        },
        "retries":       5,
        "fragment_retries": 5,
        **({"proxy": PROXY_URL} if PROXY_URL else {}),
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        # yt-dlp appends .mp3 after postprocessing — handle both cases
        final_path = audio_path if os.path.exists(audio_path) else audio_path + ".mp3"
        if not os.path.exists(final_path):
            return None, None
        return None, final_path  # no video file needed anymore
    except Exception as e:
        return None, None

# ─── Transcribe ───────────────────────────────────────────────────────────────

def transcribe(audio_path: str) -> str:
    try:
        with open(audio_path, "rb") as f:
            response = client_groq.audio.transcriptions.create(
                file=(audio_path, f.read()),
                model="whisper-large-v3",
                response_format="text",
            )
        return response if isinstance(response, str) else getattr(response, "text", str(response))
    except Exception as e:
        return f"[Transcription failed: {e}]"

# ─── Gemini sentiment ─────────────────────────────────────────────────────────

def analyze_sentiment(transcript: str, product: str, title: str) -> dict:
    safe_title      = title.replace('"', "'")
    safe_transcript = transcript[:3500].replace("\\", " ").replace("\n", " ")

    prompt = f"""You are a product review analyst. Analyze this video transcript about "{product}" and return a JSON object.

VIDEO TITLE: {safe_title}
TRANSCRIPT: {safe_transcript}

RULES:
- Return ONLY a raw JSON object. No markdown, no backticks, no explanation before or after.
- All string values must use only straight double quotes. No apostrophes inside values that could break JSON.
- Do not include newlines inside string values.

JSON FORMAT (copy this structure exactly):
{{"score": 7, "verdict": "Positive", "summary": "The reviewer liked the camera and battery life but found the price too high.", "pros": ["Great camera", "Long battery"], "cons": ["Expensive"], "confidence": "high"}}

verdict must be exactly one of: Very Positive, Positive, Mixed, Negative, Very Negative
score must be an integer from 1 to 10
confidence must be exactly one of: high, medium, low
"""
    raw = ""
    try:
        response = client_gemini.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        raw = response.text.strip()
        raw = re.sub(r"^```json\s*|^```\s*|```\s*$", "", raw, flags=re.MULTILINE).strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)
        return json.loads(raw)

    except json.JSONDecodeError:
        try:
            fix_response = client_gemini.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"Fix this broken JSON and return ONLY the corrected object:\n{raw}",
            )
            fixed = re.sub(r"^```json\s*|^```\s*|```\s*$", "", fix_response.text.strip(), flags=re.MULTILINE).strip()
            m = re.search(r"\{.*\}", fixed, re.DOTALL)
            if m:
                return json.loads(m.group(0))
        except Exception:
            pass
        return {"score": 5, "verdict": "Mixed", "summary": "Could not parse analysis.",
                "pros": [], "cons": [], "confidence": "low"}
    except Exception as e:
        return {"score": 5, "verdict": "Mixed", "summary": f"Analysis failed: {e}",
                "pros": [], "cons": [], "confidence": "low"}

# ─── Cleanup ──────────────────────────────────────────────────────────────────

def cleanup(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

# ─── Helpers ──────────────────────────────────────────────────────────────────

VERDICT_COLORS = {
    "Very Positive": "#1D9E75",
    "Positive":      "#5DCAA5",
    "Mixed":         "#EF9F27",
    "Negative":      "#D85A30",
    "Very Negative": "#E24B4A",
}

def score_color(s):
    if isinstance(s, (int, float)):
        if s >= 7: return "#1D9E75"
        if s >= 5: return "#EF9F27"
    return "#E24B4A"

def fmt_duration(secs):
    if not secs: return ""
    m, s = divmod(int(secs), 60)
    return f"{m}:{s:02d}"

# ─── Dashboard ────────────────────────────────────────────────────────────────


def gemini_master_review(product: str, results: list[dict]) -> str:
    """Single Gemini call: synthesized product review in Gemini's own voice."""
    valid = [r for r in results if isinstance(r.get("score"), int)]
    if not valid:
        return "Not enough data to generate a master review."

    avg = round(sum(r["score"] for r in valid) / len(valid), 1)

    digest_lines = []
    for i, r in enumerate(valid, 1):
        platform_label = "YouTube" if r["platform"] == "youtube" else "YouTube Short"
        pros_text = "; ".join(r.get("pros", [])) or "none mentioned"
        cons_text = "; ".join(r.get("cons", [])) or "none mentioned"
        line = "\n".join([
            "Video " + str(i) + " [" + platform_label + "] - " + r.get("title", "Unknown"),
            "  Score: " + str(r.get("score", "?")) + "/10 | Verdict: " + r.get("verdict", "?"),
            "  Summary: " + r.get("summary", ""),
            "  Pros: " + pros_text,
            "  Cons: " + cons_text,
        ])
        digest_lines.append(line)

    digest = "\n\n".join(digest_lines)

    prompt = "\n".join([
        "You are Gemini, Google's AI. You have just analyzed " + str(len(valid)) + " independent video reviews of the " + product + ".",
        "The average score across all reviews is " + str(avg) + "/10.",
        "",
        "Here is a digest of every review:",
        "",
        digest,
        "",
        "Your task: write a single, authoritative, well-structured product review synthesizing everything you have read.",
        "Write in first person as Gemini — your own informed verdict, not just summarizing what reviewers said.",
        "Be direct, opinionated, and genuinely useful to someone deciding whether to buy this product.",
        "",
        "Use exactly these markdown sections:",
        "",
        "## Gemini's Verdict on " + product,
        "",
        "### Overall impression",
        "(2-3 sentences — your top-line take)",
        "",
        "### What stands out positively",
        "(bullet points of strongest recurring praise)",
        "",
        "### What holds it back",
        "(bullet points of most common criticisms)",
        "",
        "### Who should buy it",
        "(1-2 sentences about the ideal buyer)",
        "",
        "### Who should skip it",
        "(1-2 sentences about who should look elsewhere)",
        "",
        "### Final score",
        "**X.X / 10** — one sentence justifying the score",
    ])

    try:
        response = client_gemini.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return "Could not generate master review: " + str(e)


def render_dashboard(product: str, results: list[dict]):
    valid = [r for r in results if isinstance(r.get("score"), int)]
    avg   = round(sum(r["score"] for r in valid) / len(valid), 1) if valid else 0

    st.markdown(f"## Results for **{product}**")
    c1, c2, c3 = st.columns(3)
    with c1:
        color = score_color(avg)
        st.markdown(f"""
        <div style="text-align:center;padding:16px;border:1px solid {color};border-radius:12px">
            <div style="font-size:44px;font-weight:600;color:{color}">{avg}</div>
            <div style="font-size:12px;color:var(--color-text-secondary)">average score / 10</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        yt_scores = [r["score"] for r in valid if r["platform"] == "youtube"]
        sh_scores = [r["score"] for r in valid if r["platform"] == "shorts"]
        yt_avg = round(sum(yt_scores) / len(yt_scores), 1) if yt_scores else "—"
        sh_avg = round(sum(sh_scores) / len(sh_scores), 1) if sh_scores else "—"
        st.metric("YouTube avg", f"{yt_avg}/10" if isinstance(yt_avg, float) else yt_avg)
        st.metric("Shorts avg",  f"{sh_avg}/10" if isinstance(sh_avg, float) else sh_avg)
    with c3:
        pos   = sum(1 for r in valid if r["score"] >= 7)
        neg   = sum(1 for r in valid if r["score"] <= 4)
        mixed = len(valid) - pos - neg
        st.metric("Positive",    pos)
        st.metric("Mixed / Neg", f"{mixed} / {neg}")

    st.markdown("---")

    # ── Score bar chart ───────────────────────────────────────────────────────
    st.subheader("Scores across all videos")
    plat_tag = lambda p: "YT" if p == "youtube" else "Short"
    labels   = [f"{plat_tag(r['platform'])} · {r['title'][:42]}" for r in valid]
    scores   = [r["score"] for r in valid]

    fig = go.Figure(go.Bar(
        x=scores, y=labels,
        orientation="h",
        marker_color=[score_color(s) for s in scores],
        text=[f"{s}/10" for s in scores],
        textposition="outside",
        hovertemplate="%{y}<br>Score: %{x}/10<extra></extra>",
    ))
    if valid:
        fig.add_vline(x=avg, line_dash="dash", line_color="#888780", line_width=1,
                      annotation_text=f"avg {avg}", annotation_position="top right")
    fig.update_layout(
        height=max(300, len(valid) * 52),
        margin=dict(l=10, r=60, t=20, b=20),
        xaxis=dict(range=[0, 11], title="Score"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig, width='stretch')
    st.markdown("---")

    # ── Per-video cards ───────────────────────────────────────────────────────
    st.subheader("Video-by-video breakdown")

    for section, platform in [("YouTube (long-form)", "youtube"), ("YouTube Shorts", "shorts")]:
        section_vids = [r for r in results if r["platform"] == platform]
        if not section_vids:
            continue
        st.markdown(f"#### {section}")

        for r in section_vids:
            score     = r.get("score", "—")
            verdict   = r.get("verdict", "—")
            summary   = r.get("summary", "")
            pros      = r.get("pros", [])
            cons      = r.get("cons", [])
            conf      = r.get("confidence", "")
            title     = r.get("title", "Unknown")
            url       = r.get("url", "#")
            thumbnail = r.get("thumbnail", "")
            channel   = r.get("channel", "")
            duration  = r.get("duration", 0)
            err       = r.get("error", "")
            vc        = VERDICT_COLORS.get(verdict, "#888780")
            stars     = "★" * min(int(score), 5) if isinstance(score, int) else "—"

            with st.expander(f"{stars}  {title[:65]}  |  {score}/10  ·  {verdict}"):
                col_thumb, col_meta = st.columns([1, 2])
                with col_thumb:
                    if thumbnail:
                        st.image(thumbnail, width='stretch')
                with col_meta:
                    st.markdown(f"**[{title}]({url})**")
                    if channel:
                        st.caption(f"Channel: {channel}")
                    if duration:
                        st.caption(f"Duration: {fmt_duration(duration)}")
                    st.markdown(
                        f'<span style="background:{vc}22;color:{vc};border:1px solid {vc};'
                        f'padding:2px 10px;border-radius:20px;font-size:13px;font-weight:500">'
                        f'{verdict}</span>  '
                        f'<span style="font-size:22px;font-weight:600;color:{score_color(score)}">'
                        f'{score}/10</span>',
                        unsafe_allow_html=True
                    )

                if err:
                    st.warning(f"Could not process: {err}")
                    continue

                st.markdown(f"""
                <div style="border-left:3px solid {vc};padding:10px 14px;border-radius:0 8px 8px 0;margin:10px 0">
                    {summary}
                </div>
                """, unsafe_allow_html=True)

                if pros or cons:
                    cp, cc = st.columns(2)
                    with cp:
                        if pros:
                            st.markdown("**What they liked**")
                            for p in pros:
                                st.markdown(f"+ {p}")
                    with cc:
                        if cons:
                            st.markdown("**What they criticized**")
                            for c in cons:
                                st.markdown(f"− {c}")

                st.caption(f"Analysis confidence: {conf}")

# ─── Per-video pipeline ───────────────────────────────────────────────────────

def process_video(video: dict, product: str, idx: int, total: int,
                  progress_bar, status_text) -> dict:
    title  = video["title"]
    url    = video["url"]
    prefix = f"tmp_{video['platform']}_{idx}"

    status_text.text(f"[{idx}/{total}] Downloading — {title[:55]}…")
    vid_path, audio_path = download_and_extract_audio(url, prefix)

    if not audio_path:
        return {**video, "score": None, "verdict": "—", "summary": "—",
                "pros": [], "cons": [], "confidence": "low",
                "error": "Download or audio extraction failed"}

    status_text.text(f"[{idx}/{total}] Transcribing — {title[:55]}…")
    transcript = transcribe(audio_path)

    status_text.text(f"[{idx}/{total}] Analyzing — {title[:55]}…")
    sentiment = analyze_sentiment(transcript, product, title)

    cleanup(vid_path, audio_path)
    return {**video, **sentiment}

# ─── UI ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Product Sentiment Analyzer", layout="wide")
st.title("📦 Product Sentiment Analyzer")
st.markdown("Enter a product name. The app finds independent YouTube and Shorts reviews, transcribes them, and aggregates sentiment into a score.")
st.markdown("---")

product = st.text_input(
    "Product name",
    placeholder="e.g. iPhone 17 Pro Max, Samsung Galaxy S25, Sony WH-1000XM6…"
)

with st.expander("⚙️ Settings"):
    c1, c2 = st.columns(2)
    with c1:
        n_youtube = st.slider("YouTube long-form videos", 1, 10, 5)
        n_shorts  = st.slider("YouTube Shorts",           1, 10, 5)
    with c2:
        st.caption("Long-form: independent reviews, 1–15 min.")
        st.caption("Shorts: true YouTube Shorts (≤60 s), independent creators only.")
        st.caption("Official brand channels are automatically excluded.")
        st.caption("Each video takes ~30–60 s to process.")

if product and st.button("🔍 Analyze reviews", type="primary"):
    progress_bar = st.progress(0)
    status_text  = st.empty()
    all_results  = []

    status_text.text("Searching YouTube long-form…")
    yt_videos = search_videos(product, "youtube", n_youtube)
    progress_bar.progress(5)

    status_text.text("Searching YouTube Shorts…")
    sh_videos = search_videos(product, "shorts", n_shorts)
    progress_bar.progress(10)

    all_videos = yt_videos + sh_videos
    if not all_videos:
        st.error("No videos found. Try a different product name or check your connection.")
        st.stop()

    st.info(f"Found {len(yt_videos)} YouTube + {len(sh_videos)} Shorts from independent creators. Processing {len(all_videos)} videos…")

    for i, video in enumerate(all_videos, start=1):
        result = process_video(video, product, i, len(all_videos), progress_bar, status_text)
        all_results.append(result)
        progress_bar.progress(10 + int((i / len(all_videos)) * 88))

    progress_bar.progress(100)
    status_text.empty()
    st.markdown("---")
    render_dashboard(product, all_results)


    # ── Gemini master review ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Gemini's Master Review")
    st.caption("A synthesized verdict written by Gemini after reading all the reviews above.")
    with st.spinner("Gemini is writing its master review…"):
        master = gemini_master_review(product, all_results)
    color = score_color(round(sum(r["score"] for r in all_results if isinstance(r.get("score"), int)) /
                              max(1, sum(1 for r in all_results if isinstance(r.get("score"), int))), 1))
    st.markdown(
        f'<div style="border:1px solid {color};border-radius:12px;padding:24px 28px;margin-top:8px">'
        f'{master}'
        f'</div>',
        unsafe_allow_html=True
    )