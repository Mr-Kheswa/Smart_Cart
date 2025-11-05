# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Visual theme ----------
PAGE_STYLE = """
<style>
html, body, .stApp { background: linear-gradient(180deg,#f8fbff 0%, #ffffff 100%); color: #0b2233; }
.card { background: #ffffff; border: 1px solid rgba(9,30,40,0.06); padding: 14px; border-radius: 10px; box-shadow: 0 6px 18px rgba(15,23,42,0.05); }
.app-title { font-weight:700; color:#07243a; }
.meta { color:#556b78; font-size:13px; margin-top:4px; }
.chat-inbox { max-height:360px; overflow:auto; padding:8px; }
.bubble-user { background:#eef2ff; padding:10px; border-radius:12px; margin:8px 0; text-align:right; color:#07243a;}
.bubble-assistant { background:#f0fdfa; padding:10px; border-radius:12px; margin:8px 0; text-align:left; color:#052430;}
.kv { color:#556b78; font-size:13px; }
.hint { color:#6b7f87; font-size:13px; }
.small { color:#6b7f87; font-size:12px; }
.rating { font-weight:600; color:#0b2233; }
</style>
"""

# ---------- Product data (unique descriptions/features, ratings linked here) ----------
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
            "description": (
                "A refined wireless mouse tuned for daily productivity. The ergonomic low-profile shell supports neutral wrist posture, "
                "while a precision optical sensor keeps tracking smooth across common desk surfaces. Silent-click buttons reduce distraction, "
                "and a lag-free 2.4 GHz receiver maintains stable control for mobile and desktop setups."
            ),
            "features": [
                "Ergonomic low-profile shell for comfort",
                "Precision optical sensor with adjustable DPI",
                "Silent-click buttons and smooth glides",
                "Reliable 2.4 GHz wireless receiver",
                "Smart sleep for extended battery life"
            ],
        },
        {
            "product_id": 102,
            "name": "Wireless Keyboard",
            "category": "Accessories",
            "price": 549.00,
            "rating": 4.1,
            "stock": "Limited stock",
            "tags": "wireless; keyboard; compact; low-profile; multi-device",
            "description": (
                "A compact wireless keyboard designed for clean, efficient desks. Tenkeyless layout saves space without sacrificing essentials, "
                "and low-profile scissor switches deliver crisp, consistent feedback. Multi-device pairing lets you switch between laptop, tablet, "
                "and TV in seconds—ideal for flexible workflows."
            ),
            "features": [
                "Compact tenkeyless layout to save space",
                "Low-profile scissor switches with crisp feel",
                "Multi-device pairing with quick-switch keys",
                "Dual-angle tilt feet for ergonomic setup",
                "Long-life battery with smart idle mode"
            ],
        },
        {
            "product_id": 103,
            "name": "USB-C Hub",
            "category": "Connectivity",
            "price": 699.00,
            "rating": 4.5,
            "stock": "In stock",
            "tags": "usb-c; hub; hdmi; 100w pd; sd card",
            "description": (
                "A versatile USB-C hub that turns one port into a workstation. Connect a 4K external display over HDMI, keep your laptop powered with "
                "100W Power Delivery passthrough, and attach peripherals via high-speed USB-A ports. Integrated SD/microSD readers streamline camera workflows."
            ),
            "features": [
                "HDMI 4K output for sharp external displays",
                "100W USB-C Power Delivery passthrough",
                "Two USB-A 3.0 ports for peripherals",
                "SD and microSD card readers",
                "Durable aluminum shell with heat dissipation"
            ],
        },
        {
            "product_id": 104,
            "name": "Laptop Sleeve",
            "category": "Protection",
            "price": 249.00,
            "rating": 4.0,
            "stock": "In stock",
            "tags": "sleeve; laptop; water-resistant; padded; minimalist",
            "description": (
                "A minimalist protective sleeve made for daily carry. Water-resistant fabric and dense foam padding guard against bumps, "
                "while a soft microfleece lining prevents scuffs. The slim silhouette slides easily into backpacks and briefcases."
            ),
            "features": [
                "Water-resistant exterior with durable zips",
                "High-density foam with microfleece lining",
                "Slim profile fits most bags",
                "Accessory pocket for charger and mouse",
                "Reinforced seams for commute-ready durability"
            ],
        },
        {
            "product_id": 105,
            "name": "Bluetooth Speaker",
            "category": "Audio",
            "price": 899.00,
            "rating": 4.6,
            "stock": "Limited stock",
            "tags": "bluetooth; speaker; 12h battery; waterproof; stereo",
            "description": (
                "A portable Bluetooth speaker with confident sound and practical toughness. Balanced drivers and a passive bass radiator deliver punchy audio, "
                "while a splash-resistant build and 12-hour battery life support listening indoors and outdoors."
            ),
            "features": [
                "Balanced stereo drivers with bass radiator",
                "IPX5 splash resistance for outdoor use",
                "12-hour battery life per charge",
                "Bluetooth 5.3 with quick pairing",
                "Hands-free calls via noise-reduced mic"
            ],
        },
    ]
    df = pd.DataFrame(products)
    for col in ["tags", "description", "category"]:
        df[col] = df[col].fillna("").astype(str)
    return df

# ---------- Utilities ----------
def short_summary(text: str, max_chars: int = 300) -> str:
    txt = (text or "").strip()
    if len(txt) <= max_chars:
        return txt
    # truncate at word boundary
    cut = txt[:max_chars]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "..."

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

def recommend(df: pd.DataFrame, tfidf_matrix, selected_prod: int, top_n: int = 5):
    idx_map = {pid: i for i, pid in enumerate(df["product_id"].tolist())}
    if selected_prod not in idx_map:
        return []
    i = idx_map[selected_prod]
    sim_desc = cosine_similarity(tfidf_matrix[i], tfidf_matrix).flatten()
    order = np.argsort(-sim_desc)
    results = []
    for j in order:
        pid = int(df.loc[j, "product_id"])
        if pid == selected_prod:
            continue
        results.append({
            "product_id": pid,
            "name": df.loc[j, "name"],
            "category": df.loc[j, "category"],
            "score": float(sim_desc[j]),
        })
        if len(results) >= top_n:
            break
    return results

# ---------- Assistant (AI-like; product-specific) ----------
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
        return "I can't find the selected product."
    r = row.iloc[0]
    name = r["name"]
    description = r["description"]
    rating = r["rating"]
    price = r["price"]
    stock = r["stock"]
    tags = r["tags"]
    features = PRODUCT_FEATURES_MAP.get(selected_prod, [])

    # token-based intent detection
    tokens = re.findall(r"\b[\w\-]+\b", user)
    def has_any(words): return any(w in tokens for w in words)

    if not user:
        return f"You're viewing {name}. Ask about features, usage, comparisons, benefits, price, rating, tags, stock, or recommendations."

    if has_any(["hi", "hello", "hey"]):
        return f"Hello. You're viewing {name}. What would you like to know—features, usage, or recommendations?"

    if has_any(["feature", "features", "spec", "specs", "specification", "what", "details"]):
        if features:
            return f"Key features of {name}: " + "; ".join(features) + "."
        return f"Key features of {name}: {short_summary(description, 220)}"

    if has_any(["use", "usage", "how", "setup", "install", "pair", "connect"]):
        guide = USAGE_GUIDE_MAP.get(selected_prod, short_summary(description, 200))
        return f"How to use {name}: {guide}"

    if has_any(["benefit", "benefits", "advantage", "value", "why"]):
        return f"Benefits of {name}: {BENEFIT_MAP.get(selected_prod, short_summary(description, 200))}"

    if has_any(["price", "cost", "expensive", "cheap"]):
        return f"The listed price of {name} is R{price:.2f}."

    if has_any(["rating", "stars", "review", "score"]):
        return f"Rating for {name}: {format_stars(rating)}"

    if has_any(["tag", "tags", "keywords", "labels"]):
        tag_list = ", ".join([t.strip() for t in (tags or "").split(";") if t.strip()]) or "—"
        return f"Tags for {name}: {tag_list}"

    if has_any(["stock", "available", "availability"]):
        return f"Stock availability for {name}: {stock}"

    if has_any(["recommend", "similar", "suggest", "like"]):
        return "Use 'Recommend from selected' to get similar items based on description and tags."

    # compare: extract an ID if mentioned (e.g., "compare 101 vs 105" or "compare 103")
    if has_any(["compare", "versus", "vs", "better", "alternative"]):
        ids = re.findall(r"\b(101|102|103|104|105)\b", user_raw)
        if ids:
            pid_b = int(ids[0])
            if pid_b == selected_prod:
                return "You're comparing the same product. Mention another ID like 101 or 105."
            row_b = df.loc[df["product_id"] == pid_b]
            if row_b.empty:
                return "I couldn't find that product ID. Try 101, 102, 103, 104, or 105."
            rb = row_b.iloc[0]
            return (
                f"Comparison — {name} vs {rb['name']}:\n"
                f"- Price: R{price:.2f} vs R{rb['price']:.2f}\n"
                f"- Rating: {format_stars(rating)} vs {format_stars(rb['rating'])}\n"
                f"- Category: {r['category']} vs {rb['category']}\n"
                f"- Tags: {', '.join([t.strip() for t in (tags or '').split(';') if t.strip()]) or '—'} vs "
                f"{', '.join([t.strip() for t in (rb['tags'] or '').split(';') if t.strip()]) or '—'}"
            )
        return "To compare, type: compare 101 vs 105 (or mention one ID to compare against the current)."

    # tag-aware fallback
    tag_words = [t.strip().lower() for t in (tags or "").split(";") if t.strip()]
    for w in tag_words:
        if w in tokens:
            return f"Regarding {w} on {name}: {short_summary(description, 200)}"

    return f"I don’t have an exact answer yet for that about {name}. Try asking about features, usage, comparisons, price, rating, tags, stock, or recommendations."

# ---------- App ----------
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
        "<div class='card'><div style='display:flex;align-items:center;gap:10px'>"
        "<div style='font-size:20px' class='app-title'>Smart Cart</div>"
        "<div class='meta'>Artificial intelligence powered product recommender.</div>"
        "</div></div>", unsafe_allow_html=True
    )
    st.markdown("")

    # Layout
    controls, content = st.columns([1, 2], gap="large")

    # Controls
    with controls:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Controls")
        options = df["product_id"].astype(str) + " — " + df["name"]
        choice = st.selectbox("Choose product", options)
        selected_prod = int(choice.split(" — ")[0])
        top_n = st.slider("Top N (recommendations)", 1, 8, 5)
        if st.button("Recommend from selected", key="rec_btn"):
            try:
                recs = recommend(df, tfidf, selected_prod, top_n=top_n)
                st.session_state["recs"] = recs
                st.session_state["seed"] = selected_prod
                st.session_state["chat"].append({"role": "assistant", "text": f"Recommended {len(recs)} items for {selected_prod}."})
            except Exception as e:
                st.error(f"Recommendation error: {e}")
        st.markdown("<div class='small' style='margin-top:8px'>Tip: Pick a product then ask the assistant to explain, compare, or request recommendations.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Content: Product details
    with content:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Product details")
        prod = df[df["product_id"] == selected_prod].iloc[0]
        st.markdown(f"### {prod['name']}")
        st.markdown(f"<div class='kv'>Category: {prod.get('category','')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kv'>Price: R{prod.get('price',0):.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kv'>Rating: <span class='rating'>{format_stars(prod.get('rating',''))}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kv'>Stock: {prod.get('stock','')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kv'>Tags: {', '.join([t.strip() for t in str(prod.get('tags','')).split(';') if t.strip()]) or '—'}</div>", unsafe_allow_html=True)
        st.write(short_summary(prod.get("description",""), max_chars=450))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Recommendations
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Recommendations")
        recs = st.session_state.get("recs", [])
        if not recs:
            st.info("No recommendations yet. Use the controls to generate them.")
        else:
            for rc in recs:
                row = df[df["product_id"] == rc["product_id"]]
                st.markdown(f"**{rc['name']}**  •  *{rc.get('category','')}*")
                st.markdown(f"<div class='kv'>Score: {rc.get('score',0):.3f}</div>", unsafe_allow_html=True)
                if not row.empty:
                    st.write(short_summary(row.iloc[0].get("description",""), max_chars=200))
                st.markdown("---")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Chat assistant
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Chat Assistance")
        if not st.session_state["chat"] or st.session_state.get("seed") != selected_prod:
            summary = short_summary(prod.get("description",""), max_chars=160)
            st.session_state["chat"].append({
                "role":"assistant",
                "text":f"Hello. You're viewing '{prod['name']}'. {summary} Ask about features, usage, comparisons, benefits, price, rating, tags, stock, or recommendations."
            })
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

        st.markdown("<div class='hint'>Ask about features, usage, comparisons, price, rating, tags, stock, or request recommendations.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='small' style='margin-top:8px'>© Copyright 2025 Mr Kheswa. All rights reserved.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    run_app()
