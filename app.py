# app.py
import os
from io import BytesIO
from pathlib import Path
import re

import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Visual theme and styles
# =========================
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
.rating { font-weight:600; color:#0b2233; }
</style>
"""

IMAGES_DIR = Path("data/images")  # optional local images folder

# =========================
# Data: products (unique descriptions and features)
# =========================
def build_products_df() -> pd.DataFrame:
    # Unique descriptions and features crafted here; ratings are linked in the data rows
    products = [
        {
            "product_id": 101,
            "name": "Wireless Mouse",
            "category": "Accessories",
            "price": 299.00,
            "rating": 4.3,
            "stock": "In stock",
            "tags": "wireless; mouse; ergonomic; portable; 2.4GHz",
            "image_url": "",  # you can place a local filename or remote URL here
            "features": [
                "Ergonomic low-profile shell with soft-touch finish",
                "Precision optical sensor with adjustable DPI (800–2400)",
                "Lag-free 2.4 GHz receiver with smart sleep to save power",
                "Silent-click buttons and smooth PTFE glides",
                "Ambidextrous shape with textured side grips"
            ],
            "description": (
                "A refined wireless mouse built for everyday focus and long sessions. "
                "Its ergonomic low-profile shell sits naturally in the hand, while the precision optical sensor tracks smoothly across desks and mats. "
                "Silent-click buttons reduce distraction, and the 2.4 GHz receiver delivers reliable, lag-free control. "
                "Ideal for notebooks and compact setups where comfort and portability matter."
            ),
        },
        {
            "product_id": 102,
            "name": "Wireless Keyboard",
            "category": "Accessories",
            "price": 549.00,
            "rating": 4.1,
            "stock": "Limited stock",
            "tags": "wireless; keyboard; compact; low-profile; multi-device",
            "image_url": "",
            "features": [
                "Compact tenkeyless layout to save desk space",
                "Low-profile scissor switches with crisp feedback",
                "Multi-device pairing with quick-switch keys",
                "Dual-angle tilt feet for personalized ergonomics",
                "Long-life battery with smart idle mode"
            ],
            "description": (
                "A modern wireless keyboard that balances portability and typing confidence. "
                "The compact tenkeyless frame maximizes space without sacrificing essential keys, and low-profile scissor switches keep travel short and precise. "
                "Multi-device pairing lets you jump between laptop, tablet, and TV seamlessly. "
                "Built for clean desks, mobile workflows, and lightweight productivity."
            ),
        },
        {
            "product_id": 103,
            "name": "USB-C Hub",
            "category": "Connectivity",
            "price": 699.00,
            "rating": 4.5,
            "stock": "In stock",
            "tags": "usb-c; hub; hdmi; 100w pd; sd card",
            "image_url": "",
            "features": [
                "HDMI 4K output for external displays",
                "100W USB-C Power Delivery passthrough",
                "Two USB-A 3.0 ports for peripherals",
                "SD/microSD card readers for creators",
                "Aluminum shell with thermal vents"
            ],
            "description": (
                "Expand a single USB-C port into a full desktop station. "
                "Connect an external 4K display, power your laptop with 100W passthrough, and plug in drives or peripherals at USB 3.0 speeds. "
                "Integrated SD and microSD readers streamline photo and video workflows. "
                "A robust aluminum body keeps the hub cool and travel-ready."
            ),
        },
        {
            "product_id": 104,
            "name": "Laptop Sleeve",
            "category": "Protection",
            "price": 249.00,
            "rating": 4.0,
            "stock": "In stock",
            "tags": "sleeve; laptop; water-resistant; padded; minimalist",
            "image_url": "",
            "features": [
                "Water-resistant fabric with durable zippers",
                "High-density foam padding with microfleece lining",
                "Slim profile for backpacks and briefcases",
                "Accessory pocket for charger and mouse",
                "Reinforced seams for daily commute protection"
            ],
            "description": (
                "A minimalist sleeve crafted to protect without bulk. "
                "Water-resistant fabric and dense foam padding guard against bumps, while a soft microfleece lining prevents scuffs. "
                "The slim silhouette slides into any bag, with a discreet pocket for essentials. "
                "Ideal for campus, office, and travel—clean look, dependable protection."
            ),
        },
        {
            "product_id": 105,
            "name": "Bluetooth Speaker",
            "category": "Audio",
            "price": 899.00,
            "rating": 4.6,
            "stock": "Limited stock",
            "tags": "bluetooth; speaker; 12h battery; waterproof; stereo",
            "image_url": "",
            "features": [
                "Balanced stereo drivers with passive bass radiator",
                "IPX5 splash-resistant design for outdoors",
                "12-hour battery life on a single charge",
                "Bluetooth 5.3 with quick pairing and stable range",
                "Hands-free calls with noise-reduced mic"
            ],
            "description": (
                "Portable sound with confident bass and clean mids. "
                "A splash-resistant build follows you from kitchen to park, while 12-hour battery life supports long playlists. "
                "Bluetooth 5.3 keeps connections steady, and the onboard mic handles quick calls. "
                "Compact, tough, and tuned for everyday listening."
            ),
        },
    ]
    df = pd.DataFrame(products)
    # Ensure strings for tags and description
    df["tags"] = df["tags"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["category"] = df["category"].fillna("").astype(str)
    df["image_url"] = df["image_url"].fillna("").astype(str)
    return df

# =========================
# Utilities
# =========================
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
    parts = [p.strip() for p in txt.split(".") if p.strip()]
    acc = ""
    for p in parts:
        if len(acc) + len(p) + 2 <= max_chars:
            acc += p + ". "
        else:
            break
    acc = acc.strip()
    if not acc:
        acc = (txt[:max_chars].strip() + "...")
    if not acc.endswith("."):
        acc += "."
    return acc

def format_stars(rating: float, max_stars: int = 5) -> str:
    try:
        r = float(rating)
    except Exception:
        return "No rating"
    full = int(r)
    half = 1 if (r - full) >= 0.5 else 0
    empty = max_stars - full - half
    return "★" * full + ("½" if half else "") + "☆" * empty + f"  ({r:.1f})"

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

    # 5) Last resort
    try:
        st.image(p, caption=caption, use_container_width=True)
        return
    except Exception:
        pass

    st.image(placeholder, caption=caption, use_container_width=True)
    st.markdown(f"<small class='small'>Could not load image: {p}</small>", unsafe_allow_html=True)

# =========================
# Recommendation engine
# =========================
def build_tfidf(texts: list[str]):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    X = vectorizer.fit_transform(texts)
    return vectorizer, X

def recommend(df: pd.DataFrame, tfidf_matrix, selected_prod: int, top_n: int = 5, w_tag: float = 0.6, w_desc: float = 0.4):
    # Description similarity
    idx_map = {pid: i for i, pid in enumerate(df["product_id"].tolist())}
    if selected_prod not in idx_map:
        return []
    i = idx_map[selected_prod]
    sim_desc = cosine_similarity(tfidf_matrix[i], tfidf_matrix).flatten()

    # Tag overlap (Jaccard)
    def tag_set(s): return set([t.strip().lower() for t in str(s).split(";") if t.strip()])
    base_tags = tag_set(df.loc[i, "tags"])
    tag_sims = []
    for _, row in df.iterrows():
        ts = tag_set(row["tags"])
        inter = len(base_tags & ts)
        union = len(base_tags | ts) or 1
        tag_sims.append(inter / union)
    tag_sims = np.array(tag_sims)

    # Combined score
    score = w_desc * sim_desc + w_tag * tag_sims
    order = np.argsort(-score)
    results = []
    for j in order:
        pid = int(df.loc[j, "product_id"])
        if pid == selected_prod:
            continue
        results.append({
            "product_id": pid,
            "name": df.loc[j, "name"],
            "category": df.loc[j, "category"],
            "score": float(score[j]),
        })
        if len(results) >= top_n:
            break
    return results

# =========================
# Assistant core
# =========================
def assistant_build_explanation_from_row(row: pd.Series) -> str:
    name = row.get("name", "Unknown product")
    category = row.get("category", "Unknown")
    tags = row.get("tags", "")
    description = str(row.get("description", "")).strip()
    price = row.get("price", "N/A")
    rating = row.get("rating", "N/A")
    stock = row.get("stock", "N/A")

    summary = short_summary(description, max_chars=300)
    tag_list = ", ".join([t.strip() for t in (tags or "").split(";") if t.strip()]) or "—"

    explanation = (
        f"{name}\n\n"
        f"Category: {category}\n"
        f"Price: R{price}  •  Rating: {format_stars(rating)}  •  Stock: {stock}\n"
        f"Tags: {tag_list}\n\n"
        f"Quick summary: {summary}\n\n"
        f"Full description: {description or 'No additional details available.'}\n"
    )
    return explanation

# Domain features per product for professional, AI-like replies
PRODUCT_FEATURES_MAP = {
    101: [
        "Ergonomic low-profile shell with soft-touch finish",
        "Adjustable DPI 800–2400 for precise control",
        "Silent-click buttons and smooth PTFE glides"
    ],
    102: [
        "Compact tenkeyless layout for space efficiency",
        "Low-profile scissor switches with crisp feel",
        "Multi-device pairing with quick-switch keys"
    ],
    103: [
        "HDMI 4K output for sharp external displays",
        "100W Power Delivery keeps laptops charged",
        "SD/microSD readers for fast imports"
    ],
    104: [
        "Water-resistant exterior with durable zippers",
        "High-density foam and soft microfleece lining",
        "Slim profile with accessory pocket"
    ],
    105: [
        "Balanced stereo with passive bass radiator",
        "IPX5 splash resistance for outdoor use",
        "12-hour battery for all-day listening"
    ],
}

def assistant_reply(user_text: str, df: pd.DataFrame, selected_prod: int):
    user_raw = (user_text or "").strip()
    user = user_raw.lower()
    row = df.loc[df["product_id"] == selected_prod]

    if row.empty:
        return "I can't find the selected product. Please choose another product from the left."

    r = row.iloc[0]
    name = r.get("name", "this product")
    description = str(r.get("description", "")).strip()
    tags = r.get("tags", "")
    rating = r.get("rating", "")
    price = r.get("price", "N/A")
    stock = r.get("stock", "N/A")
    features = PRODUCT_FEATURES_MAP.get(selected_prod, [])

    intents = {
        "features": ["feature", "features", "spec", "specs", "specification", "what does", "what is"],
        "usage": ["use", "how to", "how do", "setup", "install", "apply", "charge", "pair"],
        "compare": ["compare", "better", "vs", "versus", "alternative"],
        "benefit": ["benefit", "why", "good for", "why buy", "advantages", "value"],
        "price": ["price", "cost", "expensive", "cheap", "afford"],
        "rating": ["rating", "stars", "review", "reviews", "score"],
        "tags": ["tag", "tags", "keywords", "label"],
        "stock": ["stock", "available", "availability"],
        "recommend": ["recommend", "suggest", "similar", "like"],
        "greeting": ["hi", "hello", "hey", "good day"]
    }

    # Token-based intent detection to avoid substring false positives
    tokens = re.findall(r"\b[\w\-]+\b", user)
    matched = None
    for intent, keywords in intents.items():
        if any(kw in tokens for kw in keywords):
            matched = intent
            break

    # Greeting or empty
    if not user or matched == "greeting":
        return f"Hello. You're viewing {name}. Ask about features, usage, comparisons, benefits, price, rating, tags, stock, or recommendations."

    # Intent-specific replies
    if matched == "features":
        if features:
            bullet = "; ".join(features[:3])
            return f"Key features of {name}: {bullet}."
        # Fallback to first sentences of description
        sentences = [s.strip() for s in description.split(".") if s.strip()]
        if sentences:
            return f"Key features of {name}: " + " ".join(sentences[:2]) + "."
        return f"No detailed features are available for {name}."

    if matched == "usage":
        guide = {
            101: "Insert the 2.4 GHz receiver, power on the mouse, set DPI with the top button, and enable smart sleep by leaving it idle when not in use.",
            102: "Press the pairing key, select the keyboard from your device's Bluetooth menu, and use the quick-switch keys to toggle between paired devices.",
            103: "Connect the hub to your laptop’s USB-C port, attach HDMI to your display, plug power into the PD port for passthrough, and mount drives or cards as needed.",
            104: "Slide the laptop into the main compartment, zip fully, and store accessories in the outer pocket. Avoid overpacking to maintain the slim profile.",
            105: "Hold power to turn on, pair via Bluetooth 5.3, adjust volume on the speaker, and recharge after sessions to maintain the 12-hour battery."
        }.get(selected_prod, short_summary(description, max_chars=200))
        return f"How to use {name}: {guide}"

    if matched == "compare":
        ids = re.findall(r"\b\d{3}\b", user_raw)
        if len(ids) >= 1:
            try:
                pid_b = int(ids[0])
                row_b = df.loc[df["product_id"] == pid_b]
                if row_b.empty:
                    return "I couldn't find the product ID you mentioned. Try: compare 101 vs 105."
                rb = row_b.iloc[0]
                return (
                    f"Comparison — {name} vs {rb.get('name')}:\n"
                    f"- Price: R{price} vs R{rb.get('price')}\n"
                    f"- Rating: {format_stars(rating)} vs {format_stars(rb.get('rating'))}\n"
                    f"- Category: {r.get('category')} vs {rb.get('category')}\n"
                    f"- Tags: {', '.join([t.strip() for t in str(r.get('tags','')).split(';') if t.strip()]) or '—'} vs "
                    f"{', '.join([t.strip() for t in str(rb.get('tags','')).split(';') if t.strip()]) or '—'}"
                )
            except Exception:
                return "To compare, type: compare 101 vs 105 or compare 103."
        return "To compare, type: compare 101 vs 105 or mention a product ID."

    if matched == "benefit":
        value = {
            101: "Comfortable control for long sessions with quiet clicks and reliable tracking.",
            102: "Space-saving productivity with confident typing and quick device switching.",
            103: "One-port expansion for displays, power, and fast peripherals—ideal for mobile work.",
            104: "Slim, protective carry with water resistance and organized storage.",
            105: "Portable audio with steady Bluetooth and all-day battery life."
        }.get(selected_prod, "Practical value from its described features.")
        return f"Benefits of {name}: {value}"

    if matched == "price":
        return f"The listed price of {name} is R{price:.2f}."

    if matched == "rating":
        return f"Rating for {name}: {format_stars(rating)}"

    if matched == "tags":
        tag_list = ", ".join([t.strip() for t in (tags or "").split(";") if t.strip()]) or "No tags set"
        return f"Tags for {name}: {tag_list}"

    if matched == "stock":
        return f"Stock availability for {name}: {stock}"

    if matched == "recommend":
        return "Use 'Recommend from selected' to get similar items based on tags and description."

    # Tag-aware fallback
    tag_words = [t.strip().lower() for t in (tags or "").split(";") if t.strip()]
    for w in tag_words:
        if w in user:
            return f"Regarding {w} on {name}: {short_summary(description, max_chars=200)}"

    # Default fallback
    return f"I don’t have an exact answer yet for that about {name}. Try asking about features, usage, comparisons, price, rating, tags, stock, or recommendations."

# =========================
# Streamlit App
# =========================
def run_app():
    st.set_page_config(page_title="Smart Cart", layout="wide")
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)

    # Data
    try:
        df = build_products_df()
    except Exception as e:
        st.error(f"Failed to build products: {e}")
        return

    df = ensure_image_column(df)
    vect, tfidf = build_tfidf(df["description"].tolist())

    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("recs", [])
    st.session_state.setdefault("seed", None)
    st.session_state.setdefault("inbox_input", "")

    # Header
    st.markdown(
        "<div class='card'><div style='display:flex;align-items:center;gap:10px'>"
        "<div style='font-size:20px' class='app-title'>Smart Cart</div>"
        "<div class='meta'>Artificial Intelligence powered product recommender.</div>"
        "</div></div>", unsafe_allow_html=True
    )
    st.markdown("")

    controls, content, chat_col = st.columns([1, 2, 1], gap="large")

    # Controls
    with controls:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Controls")
        options = df["product_id"].astype(str) + " — " + df["name"]
        choice = st.selectbox("Choose product", options)
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
            img_val = prod.get("image_url", "")
            preview_image(img_val, caption=prod["name"])
        with cols[1]:
            st.markdown(f"### {prod['name']}")
            st.markdown(f"<div class='kv'>Category: {prod.get('category','')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='kv'>Price: R{prod.get('price',''):.2f}</div>", unsafe_allow_html=True)
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
            for rc in recs:
                row = df[df["product_id"] == rc["product_id"]]
                cols = st.columns([1, 4])
                with cols[0]:
                    if not row.empty:
                        img_val = row.iloc[0].get("image_url","")
                        filename = os.path.basename(img_val)
                        p = os.path.join("data", "images", filename) if filename else ""
                        if filename and os.path.exists(p):
                            st.image(open(p,"rb").read(), width=120, use_container_width=False)
                        else:
                            preview_image(img_val or "", caption=row.iloc[0].get("name",""), width=120)
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
        summary = short_summary(prod.get("description", ""), max_chars=160)
        st.session_state["chat"].append({
            "role": "assistant",
            "text": f"Hello. You're viewing '{prod['name']}'. {summary} Ask about features, usage, comparisons, benefits, price, rating, tags, stock, or recommendations."
        })
        st.session_state["seed"] = selected_prod

    # render chat history
    st.markdown("<div class='chat-inbox'>", unsafe_allow_html=True)
    for msg in st.session_state["chat"][-50:]:
        if msg.get("role") == "user":
            st.markdown(
                f"<div class='bubble-user'><small>{msg['text']}</small></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='bubble-assistant'><small>{msg['text']}</small></div>",
                unsafe_allow_html=True
            )
    st.markdown("</div>", unsafe_allow_html=True)

    # input and callbacks
    def _send_callback():
        user_text = st.session_state.get("inbox_input", "").strip()
        if not user_text:
            return
        st.session_state["chat"].append({"role": "user", "text": user_text})
        reply = assistant_reply(user_text, df, selected_prod)
        st.session_state["chat"].append({"role": "assistant", "text": reply})
        st.session_state["inbox_input"] = ""

    st.text_input("Type message (press Enter)", key="inbox_input", on_change=_send_callback)
    if st.button("Clear chat", key="clear_chat"):
        st.session_state["chat"] = []
        st.session_state["seed"] = None

    st.markdown(
        "<div class='hint'>If the assistant can't answer, try asking about features, usage, comparisons, price, rating, tags, stock, or request recommendations from the left.</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
    