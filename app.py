import streamlit as st
import os
import re
import json
import yt_dlp
import plotly.graph_objects as go
from dotenv import load_dotenv
from google import genai
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

load_dotenv()

# ─── API Keys ─────────────────────────────────────────────────────────────────

def get_key(name):
    try:
        val = st.secrets[name]
        if val: return val
    except Exception:
        pass
    return os.getenv(name)

gemini_key = get_key("GEMINI_API_KEY")
if not gemini_key:
    st.error("Missing API key: GEMINI_API_KEY")
    st.stop()

client_gemini = genai.Client(api_key=gemini_key)

# ─── Brand filter ─────────────────────────────────────────────────────────────

BRAND_KEYWORDS = [
    "samsung", "apple", "google", "sony", "huawei", "xiaomi", "oppo", "oneplus",
    "motorola", "nokia", "lg electronics", "microsoft", "official", "officiel",
    "corp", "inc", "ltd", "gmbh",
]

def is_brand_channel(channel: str, product: str) -> bool:
    if not channel: return False
    ch = channel.lower().strip()
    brand = product.lower().split()[0] if product else ""
    if brand and (ch == brand or ch.startswith(brand)): return True
    return any(kw in ch for kw in BRAND_KEYWORDS)

# ─── Video search ─────────────────────────────────────────────────────────────

def search_videos(query: str, platform: str, max_results: int = 5) -> list[dict]:
    if platform == "youtube":
        search_url   = f"ytsearch{max_results * 4}:{query} review -official"
        max_duration = 900
        min_duration = 60
    else:
        search_url   = f"ytsearch{max_results * 4}:{query} #shorts review"
        max_duration = 60
        min_duration = 10

    ydl_opts = {"quiet": True, "extract_flat": True, "no_warnings": True}
    results  = []

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info    = ydl.extract_info(search_url, download=False)
            entries = info.get("entries", []) if info else []
            for entry in entries:
                if not entry: continue
                duration = entry.get("duration") or 0
                if duration > max_duration or duration < min_duration: continue
                channel = (entry.get("uploader") or entry.get("channel") or "").strip()
                if is_brand_channel(channel, query): continue
                vid_id = entry.get("id", "")
                url = entry.get("url") or entry.get("webpage_url") or ""
                if not url and vid_id:
                    url = f"https://www.youtube.com/watch?v={vid_id}"
                if platform == "shorts" and vid_id:
                    url = f"https://www.youtube.com/shorts/{vid_id}"
                if not url: continue
                results.append({
                    "title":     entry.get("title", "Unknown"),
                    "url":       url,
                    "vid_id":    vid_id,
                    "duration":  duration,
                    "platform":  platform,
                    "thumbnail": f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg" if vid_id else "",
                    "channel":   channel,
                })
                if len(results) >= max_results: break
    except Exception as e:
        st.warning(f"Search error ({platform}): {e}")

    return results

# ─── Transcript fetch ─────────────────────────────────────────────────────────

def get_transcript(vid_id: str) -> str:
    """Fetch captions directly from YouTube. No download, no auth needed."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(vid_id)
        try:
            t = transcript_list.find_transcript(["en", "en-US", "en-GB"])
        except Exception:
            t = transcript_list.find_generated_transcript(
                [x.language_code for x in transcript_list]
            )
        entries = t.fetch()
        return " ".join(e.get("text", "") for e in entries).strip()
    except TranscriptsDisabled:
        return ""
    except NoTranscriptFound:
        return ""
    except Exception:
        return ""

# ─── Gemini sentiment ─────────────────────────────────────────────────────────

def analyze_sentiment(transcript: str, product: str, title: str) -> dict:
    safe_title      = title.replace('"', "'")
    safe_transcript = transcript[:4000].replace("\\", " ").replace("\n", " ")

    prompt = f"""You are a product review analyst. Analyze this video transcript about "{product}" and return a JSON object.

VIDEO TITLE: {safe_title}
TRANSCRIPT: {safe_transcript}

RULES:
- Return ONLY a raw JSON object. No markdown, no backticks, no explanation.
- Use only straight double quotes. No newlines inside string values.

JSON FORMAT:
{{"score": 7, "verdict": "Positive", "summary": "The reviewer liked the camera but found it expensive.", "pros": ["Great camera", "Long battery"], "cons": ["Expensive"], "confidence": "high"}}

verdict: one of Very Positive, Positive, Mixed, Negative, Very Negative
score: integer 1-10
confidence: high, medium, or low
"""
    raw = ""
    try:
        response = client_gemini.models.generate_content(
            model="gemini-2.5-flash", contents=prompt)
        raw = response.text.strip()
        raw = re.sub(r"^```json\s*|^```\s*|```\s*$", "", raw, flags=re.MULTILINE).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m: raw = m.group(0)
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            fix = client_gemini.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"Fix this JSON and return ONLY the corrected object:\n{raw}")
            fixed = re.sub(r"^```json\s*|^```\s*|```\s*$", "", fix.text.strip(), flags=re.MULTILINE).strip()
            m2 = re.search(r"\{.*\}", fixed, re.DOTALL)
            if m2: return json.loads(m2.group(0))
        except Exception:
            pass
        return {"score": 5, "verdict": "Mixed", "summary": "Could not parse analysis.",
                "pros": [], "cons": [], "confidence": "low"}
    except Exception as e:
        return {"score": 5, "verdict": "Mixed", "summary": f"Analysis failed: {e}",
                "pros": [], "cons": [], "confidence": "low"}

# ─── Gemini master review ─────────────────────────────────────────────────────

def gemini_master_review(product: str, results: list[dict]) -> str:
    valid = [r for r in results if isinstance(r.get("score"), int)]
    if not valid: return "Not enough data."
    avg = round(sum(r["score"] for r in valid) / len(valid), 1)

    digest_lines = []
    for i, r in enumerate(valid, 1):
        label = "YouTube" if r["platform"] == "youtube" else "YouTube Short"
        pros  = "; ".join(r.get("pros", [])) or "none"
        cons  = "; ".join(r.get("cons", [])) or "none"
        digest_lines.append("\n".join([
            f"Video {i} [{label}] - {r.get('title','?')}",
            f"  Score: {r.get('score','?')}/10 | {r.get('verdict','?')}",
            f"  Summary: {r.get('summary','')}",
            f"  Pros: {pros} | Cons: {cons}",
        ]))

    prompt = "\n".join([
        f"You are Gemini. You analyzed {len(valid)} independent reviews of {product} (avg {avg}/10).",
        "", "REVIEWS:", "", "\n\n".join(digest_lines), "",
        "Write an authoritative first-person review synthesizing all videos.",
        "Be direct and useful to a buyer. Use these exact markdown sections:",
        f"## Gemini's Verdict on {product}",
        "### Overall impression", "### What stands out positively",
        "### What holds it back", "### Who should buy it",
        "### Who should skip it", "### Final score",
        "**X.X / 10** — one sentence justifying the score",
    ])

    try:
        return client_gemini.models.generate_content(
            model="gemini-2.5-flash", contents=prompt).text.strip()
    except Exception as e:
        return f"Could not generate master review: {e}"

# ─── Helpers ──────────────────────────────────────────────────────────────────

VERDICT_COLORS = {
    "Very Positive": "#1D9E75", "Positive": "#5DCAA5",
    "Mixed": "#EF9F27", "Negative": "#D85A30", "Very Negative": "#E24B4A",
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

# ─── Per-video pipeline ───────────────────────────────────────────────────────

def process_video(video: dict, product: str, idx: int, total: int,
                  progress_bar, status_text) -> dict:
    title  = video["title"]
    vid_id = video.get("vid_id", "")

    if not vid_id:
        m = re.search(r"(?:v=|shorts/)([A-Za-z0-9_-]{11})", video.get("url", ""))
        vid_id = m.group(1) if m else ""

    if not vid_id:
        return {**video, "score": None, "verdict": "—", "summary": "No video ID found",
                "pros": [], "cons": [], "confidence": "low", "error": "No video ID"}

    status_text.text(f"[{idx}/{total}] Fetching transcript — {title[:50]}…")
    transcript = get_transcript(vid_id)

    if not transcript:
        return {**video, "score": None, "verdict": "—",
                "summary": "No transcript available for this video.",
                "pros": [], "cons": [], "confidence": "low",
                "error": "No transcript available"}

    status_text.text(f"[{idx}/{total}] Analyzing — {title[:50]}…")
    return {**video, **analyze_sentiment(transcript, product, title)}

# ─── Dashboard ────────────────────────────────────────────────────────────────

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
            <div style="font-size:12px;opacity:0.6">average score / 10</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        yt = [r["score"] for r in valid if r["platform"] == "youtube"]
        sh = [r["score"] for r in valid if r["platform"] == "shorts"]
        st.metric("YouTube avg", f"{round(sum(yt)/len(yt),1)}/10" if yt else "—")
        st.metric("Shorts avg",  f"{round(sum(sh)/len(sh),1)}/10" if sh else "—")
    with c3:
        pos = sum(1 for r in valid if r["score"] >= 7)
        neg = sum(1 for r in valid if r["score"] <= 4)
        st.metric("Positive",    pos)
        st.metric("Mixed / Neg", f"{len(valid)-pos-neg} / {neg}")

    st.markdown("---")
    st.subheader("Scores across all videos")

    labels = [f"{'YT' if r['platform']=='youtube' else 'Short'} · {r['title'][:42]}" for r in valid]
    scores = [r["score"] for r in valid]
    fig = go.Figure(go.Bar(
        x=scores, y=labels, orientation="h",
        marker_color=[score_color(s) for s in scores],
        text=[f"{s}/10" for s in scores], textposition="outside",
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
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    st.subheader("Video-by-video breakdown")
    for section, platform in [("YouTube (long-form)", "youtube"), ("YouTube Shorts", "shorts")]:
        vids = [r for r in results if r["platform"] == platform]
        if not vids: continue
        st.markdown(f"#### {section}")
        for r in vids:
            score    = r.get("score", "—")
            verdict  = r.get("verdict", "—")
            vc       = VERDICT_COLORS.get(verdict, "#888780")
            stars    = "★" * min(int(score), 5) if isinstance(score, int) else "—"
            err      = r.get("error", "")

            with st.expander(f"{stars}  {r['title'][:65]}  |  {score}/10  ·  {verdict}"):
                c_thumb, c_meta = st.columns([1, 2])
                with c_thumb:
                    if r.get("thumbnail"):
                        st.image(r["thumbnail"], use_container_width=True)
                with c_meta:
                    st.markdown(f"**[{r['title']}]({r.get('url','#')})**")
                    if r.get("channel"): st.caption(f"Channel: {r['channel']}")
                    if r.get("duration"): st.caption(f"Duration: {fmt_duration(r['duration'])}")
                    st.markdown(
                        f'<span style="background:{vc}22;color:{vc};border:1px solid {vc};'
                        f'padding:2px 10px;border-radius:20px;font-size:13px">{verdict}</span>'
                        f'  <span style="font-size:22px;font-weight:600;color:{score_color(score)}">'
                        f'{score}/10</span>', unsafe_allow_html=True)

                if err:
                    st.warning(f"Skipped: {err}")
                    continue

                st.markdown(f"""
                <div style="border-left:3px solid {vc};padding:10px 14px;border-radius:0 8px 8px 0;margin:10px 0">
                    {r.get("summary","")}
                </div>""", unsafe_allow_html=True)

                pros, cons = r.get("pros", []), r.get("cons", [])
                if pros or cons:
                    cp, cc = st.columns(2)
                    with cp:
                        if pros:
                            st.markdown("**What they liked**")
                            for p in pros: st.markdown(f"+ {p}")
                    with cc:
                        if cons:
                            st.markdown("**What they criticized**")
                            for c in cons: st.markdown(f"− {c}")
                st.caption(f"Confidence: {r.get('confidence','')}")

# ─── UI ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Product Sentiment Analyzer", layout="wide")
st.title("📦 Product Sentiment Analyzer")
st.markdown("Enter a product name. The app finds independent YouTube reviews, reads their captions, and aggregates sentiment into a score.")
st.markdown("---")

product = st.text_input("Product name",
    placeholder="e.g. iPhone 17 Pro Max, Samsung Galaxy S25, Sony WH-1000XM6…")

with st.expander("⚙️ Settings"):
    c1, c2 = st.columns(2)
    with c1:
        n_youtube = st.slider("YouTube long-form videos", 1, 10, 5)
        n_shorts  = st.slider("YouTube Shorts", 1, 10, 5)
    with c2:
        st.caption("Uses YouTube captions — no download, no bot detection.")
        st.caption("Videos without captions are skipped automatically.")
        st.caption("Only GEMINI_API_KEY required.")

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
        st.error("No videos found. Try a different product name.")
        st.stop()

    st.info(f"Found {len(yt_videos)} YouTube + {len(sh_videos)} Shorts. Processing…")

    for i, video in enumerate(all_videos, start=1):
        result = process_video(video, product, i, len(all_videos), progress_bar, status_text)
        all_results.append(result)
        progress_bar.progress(10 + int((i / len(all_videos)) * 85))

    progress_bar.progress(100)
    status_text.empty()
    st.markdown("---")
    render_dashboard(product, all_results)

    st.markdown("---")
    st.subheader("Gemini's Master Review")
    st.caption("Synthesized verdict written by Gemini after reading all reviews.")
    with st.spinner("Gemini is writing its master review..."):
        master = gemini_master_review(product, all_results)
    valid_scores = [r["score"] for r in all_results if isinstance(r.get("score"), int)]
    bc = score_color(round(sum(valid_scores)/len(valid_scores), 1) if valid_scores else 5)
    st.markdown(
        f'<div style="border:1px solid {bc};border-radius:12px;padding:24px 28px;margin-top:8px">'
        + master.replace("\n", "<br>") + '</div>',
        unsafe_allow_html=True)