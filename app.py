# app.py
import os
import re
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- fluent styling with animations --------------------
PAGE_STYLE = """
<style>
:root {
  --bg: #f6f8fb;
  --card: #ffffff;
  --text: #0f172a;
  --muted: #64748b;
  --border: rgba(2, 6, 23, 0.08);
  --accent: #2563eb;    /* Smart Cart blue */
  --accent-2: #10b981;  /* Emerald green */
  --accent-3: #f59e0b;  /* Amber for highlights */
}

html, body, .stApp {
  background: var(--bg);
  color: var(--text);
  font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

.block-space { height: 16px; }

.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
  padding: 18px;
  margin-bottom: 16px;
  transition: transform 0.22s ease, box-shadow 0.22s ease;
}
.card:hover {
  transform: translateY(-2px);
  box-shadow: 0 16px 30px rgba(15, 23, 42, 0.12);
}

.app-title {
  font-weight: 800;
  font-size: 24px;
  color: var(--accent);
  letter-spacing: 0.2px;
}
.meta {
  color: var(--muted);
  font-size: 14px;
}

/* Key/value text */
.kv {
  color: var(--muted);
  font-size: 14px;
  margin-bottom: 6px;
}
.rating {
  font-weight: 700;
  color: var(--accent);
}

/* Chat */
.chat-inbox {
  max-height: 480px;
  overflow-y: auto;
  padding: 8px;
}
.bubble-user, .bubble-assistant {
  animation: fadeInUp 280ms ease;
}
.bubble-user {
  background: linear-gradient(135deg, var(--accent) 0%, #4f86ff 100%);
  color: #ffffff;
  padding: 10px 12px;
  border-radius: 12px 12px 4px 12px;
  margin: 10px 0;
  text-align: right;
  display: inline-block;
  max-width: 92%;
  word-wrap: break-word;
  box-shadow: 0 6px 18px rgba(37, 99, 235, 0.25);
}
.bubble-assistant {
  background: linear-gradient(135deg, var(--accent-2) 0%, #34d399 100%);
  color: #ffffff;
  padding: 10px 12px;
  border-radius: 12px 12px 12px 4px;
  margin: 10px 0;
  text-align: left;
  display: inline-block;
  max-width: 92%;
  word-wrap: break-word;
  box-shadow: 0 6px 18px rgba(16, 185, 129, 0.25);
}

/* Images */
img, .stImage > img {
  border-radius: 12px;
}
.product-thumb {
  width: 100%;
  max-width: 160px;
  border-radius: 12px;
  border: 1px solid var(--border);
  box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
}

/* Recommendation item container (to avoid overlap) */
.rec-item {
  display: grid;
  grid-template-columns: 140px 1fr;
  grid-gap: 14px;
  align-items: start;
}
@media (max-width: 860px) {
  .rec-item {
    grid-template-columns: 120px 1fr;
  }
}
@media (max-width: 640px) {
  .rec-item {
    grid-template-columns: 1fr;
  }
}

/* Buttons and inputs */
.stButton > button {
  background: var(--accent);
  color: #ffffff;
  border-radius: 10px;
  border: none;
  padding: 8px 14px;
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 22px rgba(37, 99, 235, 0.25);
}
.stTextInput > div > input {
  border-radius: 10px;
  border: 1px solid var(--border);
}

/* Small text */
.small { color: #6b7280; font-size: 12px; }
.hint { color: #6b7280; font-size: 13px; }

/* Animations */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* Section headers accent underline */
.section-title {
  position: relative;
  display: inline-block;
}
.section-title::after {
  content: "";
  position: absolute;
  left: 0; bottom: -6px;
  width: 48%;
  height: 3px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2), var(--accent-3));
  border-radius: 3px;
}

/* Responsive header tweaks */
@media (max-width: 900px) {
  .app-title { font-size: 22px; }
  .card { padding: 16px; }
}
@media (max-width: 640px) {
  .app-title { font-size: 20px; }
}
</style>
"""

# -------------------- Image preview helper --------------------
IMAGES_DIR = Path("data/images")

def preview_image(path_or_url: str, caption: str = "", width: int | None = None, use_container_width: bool = True):
    """
    Preview images from paths, data/images, or URLs, with graceful fallback.
    """
    placeholder = "https://via.placeholder.com/640x400.png?text=No+Image"
    p = (path_or_url or "").strip()
    show = lambda data: st.image(data, caption=caption, use_container_width=use_container_width, width=width)

    if not p:
        return show(placeholder)

    # Direct path
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                return show(f.read())
        except Exception:
            pass

    # Relative path
    candidate = Path.cwd() / p
    if candidate.exists():
        try:
            with open(candidate, "rb") as f:
                return show(f.read())
        except Exception:
            pass

    # data/images/<filename>
    name_only = Path(p).name
    local_candidate = IMAGES_DIR / name_only
    if local_candidate.exists():
        try:
            with open(local_candidate, "rb") as f:
                return show(f.read())
        except Exception:
            pass

    # URL or final fallback
    try:
        return show(p)
    except Exception:
        return show(placeholder)

# -------------------- Catalog (IDs 101–105) with linked images --------------------
def build_products_df() -> pd.DataFrame:
    products = [
        {
            "product_id": 101,
            "name": "Wireless Mouse",
            "category": "Accessories",
            "price": 299.00,
            "rating": 4.3,
            "stock": "In stock",
            "tags": "wireless; mouse; ergonomic; portable; 2.4GHz",
            "image_url": "data/images/mouse.jpg",  # mouse
            "description": (
                "An ergonomic wireless mouse tuned for precision and comfort. The low-profile shell supports a natural wrist angle, "
                "while the optical sensor ensures smooth tracking across common desk surfaces. Silent clicks and a lag-free 2.4GHz receiver "
                "deliver distraction-free control for mobile and desktop setups."
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
            "image_url": "data/images/keyboard.jpg",  # keyboard
            "description": (
                "A compact wireless keyboard built for clean, efficient desks. Tenkeyless layout saves space without sacrificing essentials, "
                "and low-profile scissor switches deliver crisp, consistent feedback. Multi-device pairing lets you switch between laptop, tablet, "
                "and TV in seconds."
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
            "image_url": "data/images/hub.jpg",  # hub
            "description": (
                "A versatile USB-C hub that turns one port into a workstation. Connect a 4K external display over HDMI, keep your laptop powered "
                "with 100W Power Delivery passthrough, and attach peripherals via high-speed USB-A ports. SD/microSD readers streamline camera workflows."
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
            "image_url": "data/images/sleeve.jpg",  # sleeve
            "description": (
                "A minimalist protective sleeve designed for daily carry. Water-resistant fabric and dense foam padding guard against bumps, "
                "while a soft microfleece lining prevents scuffs. The slim silhouette slides easily into backpacks and briefcases."
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
            "image_url": "data/images/speaker.jpg",  # speaker
            "description": (
                "A portable Bluetooth speaker with balanced stereo drivers and a passive bass radiator. A splash-resistant build and 12-hour battery life "
                "support listening indoors and outdoors."
            ),
        },
    ]
    df = pd.DataFrame(products)
    for col in ["tags", "description", "category", "image_url"]:
        df[col] = df[col].fillna("").astype(str)
    return df

# -------------------- Utilities --------------------
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

def build_tfidf(texts: list[str]):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(texts)
    return vectorizer, X

def recommend(df: pd.DataFrame, tfidf_matrix, selected_prod: int, top_n: int = 5, w_tag: float = 0.6, w_desc: float = 0.4):
    idx_map = {pid: i for i, pid in enumerate(df["product_id"].tolist())}
    if selected_prod not in idx_map:
        return []
    i = idx_map[selected_prod]
    sim_desc = cosine_similarity(tfidf_matrix[i], tfidf_matrix).flatten()

    def tag_set(s): return set([t.strip().lower() for t in str(s).split(";") if t.strip()])
    base_tags = tag_set(df.loc[i, "tags"])
    tag_sims = []
    for _, row in df.iterrows():
        ts = tag_set(row["tags"])
        inter = len(base_tags & ts)
        union = len(base_tags | ts) or 1
        tag_sims.append(inter / union)
    tag_sims = np.array(tag_sims)

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

# -------------------- Chat assistant (conversational, progressive) --------------------
PRODUCT_FEATURES_MAP = {
    101: [
        "Ergonomic low-profile shell", "Precision optical sensor with adjustable DPI",
        "Silent-click buttons", "Lag-free 2.4GHz receiver", "Smart sleep for longer battery"
    ],
    102: [
        "Tenkeyless compact layout", "Low-profile scissor switches",
        "Multi-device quick switching", "Dual-angle tilt feet", "Long-life battery idle mode"
    ],
    103: [
        "HDMI 4K output", "100W USB-C Power Delivery passthrough",
        "Two USB-A 3.0 ports", "SD and microSD card readers", "Aluminum shell cooling"
    ],
    104: [
        "Water-resistant exterior", "Dense foam + microfleece lining",
        "Slim profile fits most bags", "Accessory pocket", "Reinforced seams"
    ],
    105: [
        "Balanced stereo + bass radiator", "IPX5 splash resistance",
        "12-hour battery life", "Bluetooth 5.3 quick pairing", "Noise-reduced mic"
    ],
}
USAGE_GUIDE_MAP = {
    101: "Plug in the receiver, power on, adjust DPI via the top button. Smart sleep saves battery when idle.",
    102: "Enter pairing mode, connect via Bluetooth, use quick-switch keys to change devices.",
    103: "Connect hub to USB-C, attach HDMI to display, plug PD for passthrough, use USB/SD as needed.",
    104: "Insert laptop, zip fully, store cables in pocket, avoid overpacking to keep profile slim.",
    105: "Power on, pair via Bluetooth 5.3, adjust volume, recharge after sessions.",
}
BENEFIT_MAP = {
    101: "Comfortable, quiet control with reliable tracking for mobile and desktop workflows.",
    102: "Space-efficient typing with crisp feedback and seamless multi-device switching.",
    103: "One-port expansion for display, power, and peripherals—ideal for portable workstations.",
    104: "Slim, protective carry that resists spills and prevents scuffs during commutes.",
    105: "Portable sound with strong battery life and easy connectivity indoors or outdoors.",
}

def assistant_reply(user_text: str, df: pd.DataFrame, selected_prod: int):
    """
    Conversational assistant:
    - Starts with greeting in UI, this function focuses on natural, specific replies.
    - Answers single-scope queries (price, rating, features, usage, etc.).
    - Encourages next-step questions instead of dumping everything.
    """
    user_raw = (user_text or "").strip()
    user = user_raw.lower()
    row = df.loc[df["product_id"] == selected_prod]
    if row.empty:
        return "I can't find the selected product. Please choose another product."

    r = row.iloc[0]
    name = r.get("name", "this product")
    description = str(r.get("description", "")).strip()
    tags = r.get("tags", "")
    rating = r.get("rating", "")
    price = r.get("price", "N/A")
    stock = r.get("stock", "N/A")
    features = PRODUCT_FEATURES_MAP.get(selected_prod, [])

    intents = {
        "features": ["feature", "features", "spec", "specs", "details"],
        "usage": ["use", "usage", "how", "setup", "install", "pair", "connect"],
        "compare": ["compare", "better", "vs", "versus", "alternative"],
        "benefit": ["benefit", "benefits", "advantage", "value", "why"],
        "price": ["price", "cost", "expensive", "cheap"],
        "rating": ["rating", "stars", "review", "score"],
        "tags": ["tag", "tags", "keywords", "labels"],
        "stock": ["stock", "available", "availability"],
        "recommend": ["recommend", "suggest", "similar", "like"],
        "greeting": ["hi", "hello", "hey", "good day"]
    }

    tokens = re.findall(r"\b[\w\-]+\b", user)
    matched = None
    for intent, kws in intents.items():
        if any(kw in tokens for kw in kws):
            matched = intent
            break

    # If user greets or asks open-ended, keep it conversational
    if matched == "greeting" or user in ["hi", "hello", "hey"]:
        return f"Hi! {name} is a solid choice. Want to hear key features, price, or how it’s used?"

    if matched == "features":
        if features:
            return f"Here are the top features of {name}: " + "; ".join(features) + ". Would you like a quick use guide?"
        sentences = [s.strip() for s in description.split(".") if s.strip()]
        if sentences:
            return f"Key details on {name}: " + " ".join(sentences[:2]) + ". Curious about setup?"
        return f"I don’t have detailed features listed for {name} yet. Want a summary?"

    if matched == "usage":
        guide = USAGE_GUIDE_MAP.get(selected_prod, short_summary(description, max_chars=200))
        return f"Setup for {name}: {guide} Want me to compare it to another product ID?"

    if matched == "compare":
        ids = re.findall(r"\b(101|102|103|104|105)\b", user_raw)
        if len(ids) >= 1:
            pid_b = int(ids[0])
            if pid_b == selected_prod:
                return "You're comparing the same product. Mention another ID like 101 or 105."
            row_b = df.loc[df["product_id"] == pid_b]
            if row_b.empty:
                return "I couldn't find that product ID. Try 101, 102, 103, 104, or 105."
            rb = row_b.iloc[0]
            return (
                f"{name} vs {rb.get('name')}:\n"
                f"- Price: R{price:.2f} vs R{rb.get('price'):.2f}\n"
                f"- Rating: {format_stars(rating)} vs {format_stars(rb.get('rating'))}\n"
                f"- Category: {r.get('category')} vs {rb.get('category')}\n"
                f"- Tags: {', '.join([t.strip() for t in str(r.get('tags','')).split(';') if t.strip()]) or '—'} vs "
                f"{', '.join([t.strip() for t in str(rb.get('tags','')).split(';') if t.strip()]) or '—'}\n"
                f"Want a recommendation based on {name}?"
            )
        return "To compare, say: compare 101 vs 105 (or mention just one ID to compare against the current)."

    if matched == "benefit":
        return f"Benefits of {name}: {BENEFIT_MAP.get(selected_prod, short_summary(description, 200))} Want the price or rating next?"

    if matched == "price":
        return f"The price of {name} is R{price:.2f}. Need the rating or stock?"

    if matched == "rating":
        return f"Rating for {name}: {format_stars(rating)}. Want to know the key features?"

    if matched == "tags":
        tag_list = ", ".join([t.strip() for t in (tags or "").split(";") if t.strip()]) or "No tags set"
        return f"Tags for {name}: {tag_list}. Interested in recommendations from this?"

    if matched == "stock":
        return f"Stock availability for {name}: {stock}. Want me to compare to another item?"

    if matched == "recommend":
        return "Use the 'Recommend from selected' button to get items similar to this one. Want me to highlight its top features while you do that?"

    # Tag-aware fallback keeps it relevant
    tag_words = [t.strip().lower() for t in (tags or "").split(";") if t.strip()]
    for w in tag_words:
        if w in tokens:
            return f"On '{w}' for {name}: {short_summary(description, max_chars=200)} Want usage tips too?"

    # Open-ended fallback
    return f"I can help with {name}'s features, usage, price, rating, stock, tags, or comparisons. Where should we start?"
    

# -------------------- App --------------------
def run_app():
    st.set_page_config(page_title="Smart Cart", layout="wide")
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)

    df = build_products_df()
    _, tfidf = build_tfidf(df["description"].tolist())

    # Session state
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("recs", [])
    st.session_state.setdefault("seed", None)
    st.session_state.setdefault("inbox_input", "")

    # Header
    st.markdown(
        "<div class='card'>"
        "<div style='display:flex;align-items:center;gap:12px'>"
        "<div class='app-title'>Smart Cart</div>"
        "<div class='meta'>Professional, AI-powered product recommender and conversational assistant.</div>"
        "</div>"
        "</div>", unsafe_allow_html=True
    )

    st.markdown("<div class='block-space'></div>", unsafe_allow_html=True)

    # Three-column layout: controls (left), content (center), chat (right)
    controls_col, content_col, chat_col = st.columns([1, 1.8, 1.2], gap="large")

    # Controls
    with controls_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<span class='section-title'><h3>Controls</h3></span>", unsafe_allow_html=True)
        choice = st.selectbox("Choose product", df["product_id"].astype(str) + " — " + df["name"])
        selected_prod = int(choice.split(" — ")[0])
        top_n = st.slider("Top N (recommendations)", 1, 8, 5)
        w_tag = st.slider("Tag weight (higher favors tag overlap)", 0.0, 1.0, 0.6)
        if st.button("Recommend from selected", key="rec_btn"):
            try:
                recs = recommend(df, tfidf, selected_prod, top_n=top_n, w_tag=w_tag, w_desc=1 - w_tag)
                st.session_state["recs"] = recs
                st.session_state["seed"] = selected_prod
                st.session_state["chat"].append({"role": "assistant", "text": f"Got it. I prepared {len(recs)} recommendation(s) for {selected_prod}. Feel free to ask about them."})
            except Exception as e:
                st.error(f"Recommendation error: {e}")
        st.markdown("<div class='small' style='margin-top:8px'>Tip: Pick a product, generate recommendations, then chat for details or comparisons.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Content: product details + recommendations
    with content_col:
        # Product details block with aligned image
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<span class='section-title'><h3>Selected product</h3></span>", unsafe_allow_html=True)
        prod = df[df["product_id"] == selected_prod].iloc[0]

        # Aligned product header
        st.markdown(
            "<div style='display:grid;grid-template-columns:220px 1fr;grid-gap:16px;align-items:start;'>",
            unsafe_allow_html=True
        )
        # Image
        img_container = st.container()
        with img_container:
            preview_image(prod.get("image_url", ""), caption=prod["name"], use_container_width=True)
        # Details
        st.markdown(
            f"<div>"
            f"<h4 style='margin:0'>{prod['name']}</h4>"
            f"<div class='kv'>Category: {prod.get('category','')}</div>"
            f"<div class='kv'>Price: R{prod.get('price',0):.2f}</div>"
            f"<div class='kv'>Rating: <span class='rating'>{format_stars(prod.get('rating',''))}</span></div>"
            f"<div class='kv'>Stock: {prod.get('stock','')}</div>"
            f"<div class='kv'>Tags: {', '.join([t.strip() for t in str(prod.get('tags','')).split(';') if t.strip()]) or '—'}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)  # end grid

        st.write(short_summary(prod.get("description",""), max_chars=520))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='block-space'></div>", unsafe_allow_html=True)

        # Recommendations — aligned, no overlap
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<span class='section-title'><h3>Recommendations</h3></span>", unsafe_allow_html=True)
        recs = st.session_state.get("recs", [])
        if not recs:
            st.info("No recommendations yet. Use the controls to generate them.")
        else:
            for rc in recs:
                row = df[df["product_id"] == rc["product_id"]]
                # A single recommendation card with controlled layout
                st.markdown("<div class='rec-item'>", unsafe_allow_html=True)
                # left: image
                if not row.empty:
                    preview_image(row.iloc[0].get("image_url",""), caption="", width=140, use_container_width=False)
                else:
                    st.image("https://via.placeholder.com/160x120.png?text=IMG", width=140, use_container_width=False)
                # right: text
                if not row.empty:
                    st.markdown(f"**{rc['name']}**  •  *{rc.get('category','')}*")
                    st.markdown(f"<div class='kv'>Match score: {rc.get('score',0):.3f}</div>", unsafe_allow_html=True)
                    st.write(short_summary(row.iloc[0].get("description",""), max_chars=240))
                else:
                    st.markdown(f"**{rc['name']}**")
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("<div style='height:8px;border-bottom:1px dashed rgba(2,6,23,0.08)'></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Chat assistant on the right-hand side
    with chat_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<span class='section-title'><h3>Chat assistance</h3></span>", unsafe_allow_html=True)

        # greet when empty or product changed — natural conversation start
        if st.session_state.get("seed") != selected_prod:
            st.session_state["chat"] = []  # reset convo on product change
            st.session_state["seed"] = selected_prod
        if not st.session_state["chat"]:
            st.session_state["chat"].append({
                "role": "assistant",
                "text": f"Hi there 👋 Welcome to Smart Cart. You're viewing **{prod['name']}**. What would you like to know—features, price, or how it's used?"
            })

        # render chat history
        st.markdown("<div class='chat-inbox'>", unsafe_allow_html=True)
        for msg in st.session_state["chat"][-120:]:
            if msg.get("role") == "user":
                st.markdown(f"<div class='bubble-user'><small>{msg['text']}</small></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bubble-assistant'><small>{msg['text']}</small></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # input + callbacks (progressive, conversational tone)
        def _send_callback():
            user_text = st.session_state.get("inbox_input", "").strip()
            if not user_text:
                return
            st.session_state["chat"].append({"role": "user", "text": user_text})
            reply = assistant_reply(user_text, df, selected_prod)
            st.session_state["chat"].append({"role": "assistant", "text": reply})
            st.session_state["inbox_input"] = ""

        st.text_input("Type message (press Enter)", key="inbox_input", on_change=_send_callback)
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Clear chat", key="clear_chat"):
                st.session_state["chat"] = []
                st.session_state["seed"] = selected_prod
                st.session_state["chat"].append({
                    "role": "assistant",
                    "text": f"Hi again 👋 Let’s chat about **{prod['name']}**. Want key features, price, or a quick setup guide?"
                })
        with c2:
            st.markdown("<div class='hint'>Ask me to explain features, compare IDs (e.g., 101 vs 105), or check price, rating, stock.</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("<div class='small' style='margin-top:8px'>© 2025 Smart Cart. All rights reserved.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    run_app()
