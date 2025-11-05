# app.py
import os
import re
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Styles (professional + responsive for Smart Cart) ----------------
PAGE_STYLE = """
<style>
:root {
  --bg: #f9fafb;
  --card: #ffffff;
  --muted: #6b7280;
  --accent: #2563eb;   /* Smart Cart blue */
  --accent-2: #10b981; /* Emerald green */
}

html, body, .stApp {
  background: var(--bg);
  font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  color: #111827;
  margin: 0;
  padding: 0;
}

/* Cards */
.card {
  background: var(--card);
  border: 1px solid rgba(0,0,0,0.06);
  padding: 16px;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.06);
  margin-bottom: 16px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.card:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,0.08); }

/* Header */
.app-title { font-weight: 700; font-size: 22px; color: var(--accent); }
.meta { color: var(--muted); font-size: 14px; }

/* Key-value text */
.kv { color: var(--muted); font-size: 14px; margin-bottom: 6px; }
.rating { font-weight: 600; color: var(--accent); }

/* Chat */
.chat-inbox { max-height: 360px; overflow-y: auto; padding: 8px; }
.bubble-user {
  background: var(--accent);
  color: #fff;
  padding: 10px;
  border-radius: 12px 12px 4px 12px;
  margin: 8px 0;
  text-align: right;
  display: inline-block;
  max-width: 80%;
}
.bubble-assistant {
  background: var(--accent-2);
  color: #fff;
  padding: 10px;
  border-radius: 12px 12px 12px 4px;
  margin: 8px 0;
  text-align: left;
  display: inline-block;
  max-width: 80%;
}

/* Images */
img, .stImage > img { border-radius: 10px; }
.product-thumb {
  width: 100%;
  max-width: 160px;
  border-radius: 10px;
  border: 1px solid rgba(0,0,0,0.08);
}

/* Inputs */
.stButton > button {
  background: var(--accent);
  color: #fff;
  border-radius: 8px;
  border: none;
}
.stTextInput > div > input { border-radius: 8px; }

.small { color:#6b7f87; font-size:12px; }
.hint { color:#6b7f87; font-size:13px; }

/* Responsive */
@media (max-width: 768px) {
  .app-title { font-size: 18px; }
  .card { padding: 12px; }
  .bubble-user, .bubble-assistant { max-width: 100%; font-size: 14px; }
  .product-thumb { max-width: 120px; }
}
</style>
"""

# ---------------- Helpers for images (keeps your preview behavior) ----------------
IMAGES_DIR = Path("data/images")

def preview_image(path_or_url: str, caption: str = "", width: int | None = None, use_container_width: bool = True):
    placeholder = "https://via.placeholder.com/600x400.png?text=No+Image"
    p = (path_or_url or "").strip()
    show = lambda data: st.image(data, caption=caption, use_container_width=use_container_width, width=width)

    if not p:
        return show(placeholder)

    # direct path
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                return show(f.read())
        except Exception:
            pass

    # relative path
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

    # URL or fallback
    try:
        return show(p)
    except Exception:
        return show(placeholder)

# ---------------- Data: build in-app catalog (IDs 101–105) ----------------
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
            "image_url": "data/images/wireless_mouse.jpg",
            "description": (
                "A refined wireless mouse tuned for daily productivity. The ergonomic low-profile shell supports neutral wrist posture, "
                "while a precision optical sensor keeps tracking smooth across common desk surfaces. Silent-click buttons reduce distraction, "
                "and a lag-free 2.4 GHz receiver maintains stable control for mobile and desktop setups."
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
            "image_url": "data/images/wireless_keyboard.jpg",
            "description": (
                "A compact wireless keyboard designed for clean, efficient desks. Tenkeyless layout saves space without sacrificing essentials, "
                "and low-profile scissor switches deliver crisp, consistent feedback. Multi-device pairing lets you switch between laptop, tablet, "
                "and TV in seconds—ideal for flexible workflows."
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
            "image_url": "data/images/usb_c_hub.jpg",
            "description": (
                "A versatile USB-C hub that turns one port into a workstation. Connect a 4K external display over HDMI, keep your laptop powered with "
                "100W Power Delivery passthrough, and attach peripherals via high-speed USB-A ports. Integrated SD/microSD readers streamline camera workflows."
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
            "image_url": "data/images/laptop_sleeve.jpg",
            "description": (
                "A minimalist protective sleeve made for daily carry. Water-resistant fabric and dense foam padding guard against bumps, "
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
            "image_url": "data/images/bluetooth_speaker.jpg",
            "description": (
                "A portable Bluetooth speaker with confident sound and practical toughness. Balanced drivers and a passive bass radiator deliver punchy audio, "
                "while a splash-resistant build and 12-hour battery life support listening indoors and outdoors."
            ),
        },
    ]
    df = pd.DataFrame(products)
    for col in ["tags", "description", "category", "image_url"]:
        df[col] = df[col].fillna("").astype(str)
    return df

# ---------------- Utilities (from your “second code”, polished) ----------------
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

# ---------------- Recommendation engine (TF-IDF on descriptions) ----------------
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

    # Tag Jaccard
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

# ---------------- Chat assistant (AI-like, product-specific to IDs 101–105) ----------------
PRODUCT_FEATURES_MAP = {
    101: [
        "Ergonomic low-profile shell for comfort",
        "Adjustable DPI optical tracking",
        "Silent-click buttons and smooth glides"
    ],
    102: [
        "Space-saving tenkeyless layout",
        "Low-profile scissor switches",
        "Multi-device quick switching"
    ],
    103: [
        "HDMI 4K external display output",
        "100W USB-C Power Delivery passthrough",
        "SD/microSD reader support"
    ],
    104: [
        "Water-resistant exterior",
        "Dense foam and microfleece lining",
        "Slim profile with accessory pocket"
    ],
    105: [
        "Balanced stereo with bass radiator",
        "IPX5 splash resistance",
        "12-hour battery life"
    ],
}
USAGE_GUIDE_MAP = {
    101: "Plug in the USB receiver, power on the mouse, adjust DPI with the top button, and let smart sleep save battery when idle.",
    102: "Enter pairing mode, select the keyboard from Bluetooth, then use quick-switch keys to toggle between paired devices.",
    103: "Connect the hub to USB-C, attach HDMI to your display, plug power into PD for passthrough, and use USB/SD ports as needed.",
    104: "Insert the laptop gently, zip fully, and store accessories in the outer pocket. Avoid overpacking to keep the slim silhouette.",
    105: "Power on, pair via Bluetooth 5.3, adjust volume on-device, and recharge after sessions to maintain battery longevity.",
}
BENEFIT_MAP = {
    101: "Comfortable, quiet control with dependable tracking for mobile and desktop workflows.",
    102: "Space-efficient typing with crisp feedback and seamless device switching.",
    103: "One-port expansion for display, power, and peripherals—ideal for portable workstations.",
    104: "Slim, protective carry that resists spills and prevents scuffs during commutes.",
    105: "Portable sound with strong battery life and reliable connectivity indoors or outdoors.",
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
        "features": ["feature", "features", "spec", "specs", "specification", "details"],
        "usage": ["use", "usage", "how", "how to", "setup", "install", "pair", "connect"],
        "compare": ["compare", "better", "vs", "versus", "alternative"],
        "benefit": ["benefit", "benefits", "advantage", "value", "why"],
        "price": ["price", "cost", "expensive", "cheap"],
        "rating": ["rating", "stars", "review", "score"],
        "tags": ["tag", "tags", "keywords", "labels"],
        "stock": ["stock", "available", "availability"],
        "recommend": ["recommend", "suggest", "similar", "like"],
        "greeting": ["hi", "hello", "hey", "good day"]
    }

    # token-based detection
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
            return f"Key features of {name}: " + "; ".join(features) + "."
        sentences = [s.strip() for s in description.split(".") if s.strip()]
        if sentences:
            return f"Key features of {name}: " + " ".join(sentences[:2]) + "."
        return f"No detailed features are available for {name}."

    if matched == "usage":
        guide = USAGE_GUIDE_MAP.get(selected_prod, short_summary(description, max_chars=200))
        return f"How to use {name}: {guide}"

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
                f"Comparison — {name} vs {rb.get('name')}:\n"
                f"- Price: R{price:.2f} vs R{rb.get('price'):.2f}\n"
                f"- Rating: {format_stars(rating)} vs {format_stars(rb.get('rating'))}\n"
                f"- Category: {r.get('category')} vs {rb.get('category')}\n"
                f"- Tags: {', '.join([t.strip() for t in str(r.get('tags','')).split(';') if t.strip()]) or '—'} vs "
                f"{', '.join([t.strip() for t in str(rb.get('tags','')).split(';') if t.strip()]) or '—'}"
            )
        return "To compare, type: compare 101 vs 105 (or mention one ID to compare against the current)."

    if matched == "benefit":
        return f"Benefits of {name}: {BENEFIT_MAP.get(selected_prod, short_summary(description, 200))}"

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
        if w in tokens:
            return f"Regarding {w} on {name}: {short_summary(description, max_chars=200)}"

    # Default fallback
    return f"I don’t have an exact answer yet for that about {name}. Try asking about features, usage, comparisons, price, rating, tags, stock, or recommendations."

# ---------------- App (from your “third code”, unified and fixed) ----------------
def run_app():
    st.set_page_config(page_title="Smart Cart", layout="wide")
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)

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
    st.markdown("<div class='card'><div style='display:flex;align-items:center;gap:10px'><div style='font-size:20px' class='app-title'>Smart Cart</div><div class='meta'>Artificial Intelligence powered product recommender.</div></div></div>", unsafe_allow_html=True)
    st.markdown("")

    controls, content = st.columns([1, 2], gap="large")

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
            for r in recs:
                rc = r
                cols = st.columns([1, 4])
                with cols[0]:
                    row = df[df["product_id"] == rc["product_id"]]
                    if not row.empty:
                        img_val = row.iloc[0].get("image_url","")
                        if img_val:
                            preview_image(img_val, caption=row.iloc[0].get("name",""), width=120, use_container_width=False)
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

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Chat assistant (real-time)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Chat Assistance")
        # greet when empty or product changed
        if not st.session_state["chat"] or st.session_state.get("seed") != selected_prod:
            summary = short_summary(prod.get("description",""), max_chars=160)
            st.session_state["chat"].append({"role":"assistant","text":f"Hello. Let's chat about '{prod['name']}'. {summary} Ask about features, usage, comparisons, benefits, price, rating, tags, stock, or recommendations."})
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
        st.markdown("<div class='hint'>If the assistant can't answer, try asking about features, usage, comparisons, price, rating, tags, stock, or request recommendations from the left.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    run_app()
