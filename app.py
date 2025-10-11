# app.py
# libraries import for argument parsing, data handling, numerical operations, and text similarity analysis
import os
from io import BytesIO
from pathlib import Path

import streamlit as st
import pandas as pd
import requests

from main import load_df, build_tfidf, recommend

DATA_PATH = "products.csv"
IMAGES_DIR = Path("products.csv/image_url")  # local images folder

# ---------- Visual theme ----------
PAGE_STYLE = """
<style>
:root{
  --bg:#f7fbff; --card:#ffffff; --muted:#556b78; --accent:#6d28d9; --accent-2:#0ea5a4;
}
html, body, .stApp { background: linear-gradient(180deg,#f8fbff 0%, #ffffff 100%); color: #0b2233; }
.card { background: var(--card); border: 1px solid rgba(9,30,40,0.04); padding: 14px; border-radius: 10px; box-shadow: 0 6px 18px rgba(15,23,42,0.05); }
.app-title { font-weight:700; color:#07243a; }
.meta { color: var(--muted); font-size:13px; margin-top:4px; }
.chat-inbox { max-height:360px; overflow:auto; padding:8px; }
.bubble-user { background:#eef2ff; padding:10px; border-radius:12px; margin:8px 0; text-align:right; color:#07243a;}
.bubble-assistant { background:#f0fdfa; padding:10px; border-radius:12px; margin:8px 0; text-align:left; color:#052430;}
.kv { color: var(--muted); font-size:13px; }
.hint { color:#6b7f87; font-size:13px; }
.small { color:#6b7f87; font-size:12px; }
</style>
"""

# ---------- Image helper ----------
@st.cache_data(show_spinner=False)
def fetch_image_bytes_from_url(url: str, timeout: int = 6):
    try:
        resp = requests.get(url, timeout=timeout, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        if not ct.startswith("image/"):
            return None
        return resp.content
    except Exception:
        return None

def preview_image(path_or_url: str, caption: str = "", width: int | None = None):
    """
    Display an image. Supports:
      - Local relative paths like data/images/mouse.jpg
      - Absolute local paths like C:/full/path/to/mouse.jpg
      - Remote URLs (http/https)
    Uses use_container_width to avoid deprecation warnings.
    """
    placeholder = "https://via.placeholder.com/600x400.png?text=No+Image"
    p = (path_or_url or "").strip()
    if not p:
        st.image(placeholder, caption=caption, use_container_width=True)
        return

    # 1) If path exists as given
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                st.image(f.read(), caption=caption, use_container_width=True)
                return
        except Exception:
            pass

    # 2) Resolve relative to cwd
    candidate = Path.cwd() / p
    if candidate.exists():
        try:
            with open(candidate, "rb") as f:
                st.image(f.read(), caption=caption, use_container_width=True)
                return
        except Exception:
            pass

    # 3) If value is filename only, try data/images/<filename>
    name_only = Path(p).name
    local_candidate = IMAGES_DIR / name_only
    if local_candidate.exists():
        try:
            with open(local_candidate, "rb") as f:
                st.image(f.read(), caption=caption, use_container_width=True)
                return
        except Exception:
            pass

    # 4) Remote URL fetch
    if p.startswith("http://") or p.startswith("https://"):
        img_bytes = fetch_image_bytes_from_url(p)
        if img_bytes:
            try:
                st.image(BytesIO(img_bytes), caption=caption, use_container_width=True)
                return
            except Exception:
                pass

    # 5) Last resort: direct st.image attempt
    try:
        st.image(p, caption=caption, use_container_width=True)
        return
    except Exception:
        pass

    # fallback
    st.image(placeholder, caption=caption, use_container_width=True)
    st.markdown(f"<small class='small'>Could not load image: {p} — tried given path, cwd/{p}, {IMAGES_DIR}/{name_only}, and URL fetch.</small>", unsafe_allow_html=True)

# ---------- Utilities ----------
def ensure_image_column(df: pd.DataFrame) -> pd.DataFrame:
    if "image_url" not in df.columns:
        df["image_url"] = ""
    else:
        df["image_url"] = df["image_url"].fillna("")
    return df

def short_summary(text: str, max_chars: int = 300) -> str:
    txt = (text or "").strip()
    if len(txt) <= max_chars:
        return txt
    parts = txt.split(".")
    acc = ""
    for p in parts:
        if not p.strip():
            continue
        if len(acc) + len(p) + 1 <= max_chars:
            acc += (p.strip() + ". ")
        else:
            break
    return acc.strip() if acc else txt[:max_chars].strip() + "..."

def format_stars(rating: float, max_stars: int = 5) -> str:
    try:
        r = float(rating)
    except Exception:
        return "No rating"
    full = int(r)
    half = 1 if (r - full) >= 0.5 else 0
    empty = max_stars - full - half
    return "★" * full + ("½" if half else "") + "☆" * empty + f"  ({r:.1f})"

def assistant_build_explanation_from_row(row: pd.Series) -> str:
    name = row.get("name", "Unknown product")
    category = row.get("category", "Unknown")
    tags = row.get("tags", "")
    description = str(row.get("description", "")).strip()
    price = row.get("price", "")
    rating = row.get("rating", "")
    stock = row.get("stock", "")
    summary = short_summary(description, max_chars=300)
    tag_list = ", ".join([t.strip() for t in (tags or "").split(";") if t.strip()]) or "—"
    explanation = (
        f"{name}\n\n"
        f"Category: {category}\n"
        f"Price: {price}  •  Rating: {rating}  •  Stock: {stock}\n"
        f"Tags: {tag_list}\n\n"
        f"Quick summary: {summary}\n\n"
        f"Full description: {description or 'No additional details available.'}\n"
    )
    return explanation

def assistant_reply(user_text: str, df: pd.DataFrame, selected_prod: int):
    user = (user_text or "").strip().lower()
    row = df.loc[df["product_id"] == selected_prod]
    if row.empty:
        return "I can't find the selected product. Please choose another product from the left."
    r = row.iloc[0]
    name = r.get("name", "this product")
    description = str(r.get("description", "")).strip()
    tags = r.get("tags", "")
    rating = r.get("rating", "")

    intents = {
        "features": ["feature", "features", "what does", "what is", "spec", "specs", "specification"],
        "usage": ["use", "how to", "how do", "setup", "install", "apply", "charge"],
        "compare": ["compare", "better", "vs ", "versus", "alternative"],
        "benefit": ["benefit", "why", "good for", "why buy", "advantages"],
        "price": ["price", "cost", "expensive", "cheap"],
        "rating": ["rating", "stars", "review", "reviews"],
        "tags": ["tag", "tags", "keywords", "label"],
        "recommend": ["recommend", "suggest", "similar"],
        "greeting": ["hi", "hello", "hey", "hey,"]
    }
    matched = None
    for intent, keywords in intents.items():
        for kw in keywords:
            if kw in user:
                matched = intent
                break
        if matched:
            break

    if not user or matched == "greeting":
        return f"Hello. Let's chat about '{name}'. Ask about features, usage, benefits, tags or rating."
    if matched == "features":
        sentences = [s.strip() for s in description.split(".") if s.strip()]
        reply = "Features: "
        if sentences:
            reply += " ".join(sentences[:2]) + (". " if len(sentences) >= 2 else " ")
        else:
            reply += "No detailed features available in the description."
        return reply
    if matched == "usage":
        return f"How to use '{name}': {short_summary(description, max_chars=200)}"
    if matched == "compare":
        return "Use the left controls to generate recommendations, then ask me to compare two product IDs."
    if matched == "benefit":
        return f"Benefits: This product delivers practical value through {tags.replace(';',', ') or 'its described features'}."
    if matched == "price":
        return f"The listed price is {r.get('price','unknown')}."
    if matched == "rating":
        return f"Rating: {format_stars(rating)}"
    if matched == "tags":
        tag_list = ", ".join([t.strip() for t in (tags or "").split(";") if t.strip()]) or "No tags set"
        return f"Tags for this product: {tag_list}"
    if matched == "recommend":
        return "Use 'Recommend from selected' on the left to get similar items."

    tag_words = [t.strip().lower() for t in (tags or "").split(";") if t.strip()]
    for w in tag_words:
        if w in user:
            return f"I see you're asking about '{w}'. {short_summary(description, max_chars=200)}"

    return "I don't know the exact answer yet. Ask about features, usage, rating, tags, or request recommendations."

# ---------- App ----------
def run_app():
    st.set_page_config(page_title="Smart Cart", layout="wide")
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)

    try:
        df = load_df(DATA_PATH)
    except Exception as e:
        st.error(f"Failed to load {DATA_PATH}: {e}")
        return

    df = ensure_image_column(df)
    vect, tfidf = build_tfidf(df["description"].tolist())

    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("recs", [])
    st.session_state.setdefault("seed", None)
    st.session_state.setdefault("inbox_input", "")

    # header
    st.markdown("<div class='card'><div style='display:flex;align-items:center;gap:10px'><div style='font-size:20px' class='app-title'>Smart Cart</div><div class='meta'>Artificial Intelligence powered product recommender.</div></div></div>", unsafe_allow_html=True)
    st.markdown("")

    controls, content, chat_col = st.columns([1, 2, 1], gap="large")

    # Controls
    with controls:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Controls")
        choice = st.selectbox("Choose product", df["product_id"].astype(str) + " — " + df["name"])
        selected_prod = int(choice.split(" — ")[0])
        top_n = st.slider("Top N (recommendations)", 1, 8, 5)
        w_tag = st.slider("Tag weight (higher favors tag overlap)", 0.0, 1.0, 0.6)
        if st.button("Recommend from selected", key="rec_btn"):
            try:
                recs = recommend(df, tfidf, selected_prod, top_n=top_n, w_tag=w_tag, w_desc=1 - w_tag)
                st.session_state["recs"] = recs
                st.session_state["seed"] = selected_prod
                st.session_state["chat"].append({"role": "assistant", "text": f"Recommended {len(recs)} items for {selected_prod}."})
            except Exception as e:
                st.error(f"Recommendation error: {e}")
        st.markdown("<div class='small' style='margin-top:8px'>Tip: Pick a product then ask the assistant to explain or request recommendations.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Content
    with content:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Product details")
        prod = df[df["product_id"] == selected_prod].iloc[0]
        cols = st.columns([1, 2])
        with cols[0]:
            # show image from image_url supports local or relative using preview_image helper
            img_val = prod.get("image_url", "")
            preview_image(img_val, caption=prod["name"])
        with cols[1]:
            st.markdown(f"### {prod['name']}")
            st.markdown(f"<div class='kv'>Category: {prod.get('category','')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='kv'>Price: R{prod.get('price','')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='kv'>Rating: <span class='rating'>{format_stars(prod.get('rating',''))}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='kv'>Tags: {', '.join([t.strip() for t in str(prod.get('tags','')).split(';') if t.strip()]) or '—'}</div>", unsafe_allow_html=True)
            st.write(short_summary(prod.get("description",""), max_chars=450))
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Recommendations")
        recs = st.session_state.get("recs", [])
        if not recs:
            st.info("No recommendations yet. Use the controls to generate them.")
        else:
            for r in recs:
                rc = r
                cols = st.columns([1, 4])
                with cols[0]:
                    row = df[df["product_id"] == rc["product_id"]]
                    if not row.empty:
                        img_val = row.iloc[0].get("image_url","")
                        filename = os.path.basename(img_val)
                        p = os.path.join("data", "images", filename)
                        if os.path.exists(p):
                            # fixed-size thumbnail
                            st.image(open(p,"rb").read(), width=120, use_container_width=False)
                        else:
                            st.image("https://via.placeholder.com/160x120.png?text=IMG", width=120, use_container_width=False)
                    else:
                        st.image("https://via.placeholder.com/160x120.png?text=IMG", width=120, use_container_width=False)
                with cols[1]:
                    st.markdown(f"**{rc['name']}**  •  *{rc.get('category','')}*")
                    st.markdown(f"<div class='kv'>Score: {rc.get('score',0):.3f}</div>", unsafe_allow_html=True)
                    if not row.empty:
                        st.write(short_summary(row.iloc[0].get("description",""), max_chars=200))
                st.markdown("---")
    
        st.markdown("<div class='small' style='margin-top:8px'>© Copyright 2025 Mr Kheswa. All rights reserved.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Chat assistant
    with chat_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Chat Assistance")
        # greet when empty or product changed
        if not st.session_state["chat"] or st.session_state.get("seed") != selected_prod:
            st.session_state["chat"].append({"role":"assistant","text":f"Hello. Let's chat about '{prod['name']}'. Ask about features, usage, comparisons, or request recommendations."})
            st.session_state["seed"] = selected_prod

        st.markdown("<div class='chat-inbox'>", unsafe_allow_html=True)
        for msg in st.session_state["chat"][-50:]:
            if msg.get("role") == "user":
                st.markdown(f"<div class='bubble-user'><small>{msg['text']}</small></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bubble-assistant'><small>{msg['text']}</small></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        def _send_callback():
            user_text = st.session_state.get("inbox_input", "").strip()
            if not user_text:
                return
            st.session_state["chat"].append({"role":"user","text":user_text})
            reply = assistant_reply(user_text, df, selected_prod)
            st.session_state["chat"].append({"role":"assistant","text":reply})
            st.session_state["inbox_input"] = ""

        st.text_input("Type message (press Enter)", key="inbox_input", on_change=_send_callback)
        if st.button("Clear chat", key="clear_chat"):
            st.session_state["chat"] = []
            st.session_state["seed"] = None
        st.markdown("<div class='hint'>If the assistant can't answer, try asking about features, usage, or request recommendations from the left.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    run_app()
