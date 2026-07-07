"""
streamlit_app.py
----------------
Streamlit frontend for the Car Price Prediction FastAPI backend.

Run from the project root:
    streamlit run streamlit_app.py

The FastAPI backend must be running separately:
    uvicorn app:app --reload

To point this app at a deployed backend instead of localhost, set the
BASE_URL environment variable, e.g.:
    export BASE_URL=https://your-fastapi-service.onrender.com
"""

import os

import matplotlib.pyplot as plt
import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
# Switch between local and deployed backend by setting the BASE_URL
# environment variable — no code changes needed.
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
API_TIMEOUT = 10  # seconds before giving up on the request

# --------------------------------------------------------------------------- #
# PAGE SETUP
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
# CUSTOM CSS
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    .app-subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.05rem;
        margin-top: -0.6rem;
        margin-bottom: 1.8rem;
    }
    .section-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #374151;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
    }
    div.stButton > button {
        width: 100%;
        height: 3.2rem;
        font-size: 1.1rem;
        font-weight: 700;
        border-radius: 12px;
        background-color: #16a34a;
        color: white;
        border: none;
        margin-top: 0.5rem;
    }
    div.stButton > button:hover {
        background-color: #15803d;
        color: white;
    }
    .result-card {
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
        color: white;
        padding: 2rem 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 6px 20px rgba(22, 163, 74, 0.28);
    }
    .result-card .result-label {
        font-size: 0.95rem;
        opacity: 0.88;
        margin-bottom: 0.3rem;
    }
    .result-card .result-price {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .result-card .result-note {
        font-size: 0.82rem;
        opacity: 0.75;
        margin-top: 0.5rem;
    }
    .divider { margin: 1.5rem 0; border-top: 1px solid #e5e7eb; }

    /* Make st.metric look good on wide/web layouts */
    div[data-testid="stMetric"] {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 14px;
        padding: 1.2rem 1rem;
        text-align: center;
    }
    div[data-testid="stMetricLabel"] {
        justify-content: center;
        font-size: 0.95rem;
        color: #166534;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 800;
        color: #15803d;
    }
    div[data-testid="stMetricDelta"] {
        justify-content: center;
    }

    /* Responsive tweak for smaller / mobile web widths */
    @media (max-width: 640px) {
        div[data-testid="stMetricValue"] { font-size: 1.7rem; }
        .result-card .result-price { font-size: 2.1rem; }
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #
def format_inr(amount: float) -> str:
    """
    Format a number in Indian Rupee style with lakh/crore comma placement.
    e.g. 530444 -> ₹5,30,444
    """
    amount = int(round(amount))
    s = str(amount)

    # Indian number system: last 3 digits, then groups of 2
    if len(s) <= 3:
        return f"₹{s}"

    last3    = s[-3:]
    remaining = s[:-3]
    parts    = []
    while len(remaining) > 2:
        parts.append(remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        parts.append(remaining)
    parts.reverse()
    return "₹" + ",".join(parts) + "," + last3


def call_api(endpoint: str, payload: dict) -> dict:
    """
    Central helper for all FastAPI backend calls.

    Args:
        endpoint: API path, e.g. "/predict" or "/predict-with-explanation"
        payload: JSON-serializable request body

    Returns:
        Parsed JSON response as a dict.

    Raises:
        requests.exceptions.RequestException: on network/connection/timeout errors.
        requests.exceptions.HTTPError: if the API returns a non-2xx status code.
    """
    url = f"{BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    response = requests.post(url, json=payload, timeout=API_TIMEOUT)
    response.raise_for_status()
    return response.json()


def call_predict_api(payload: dict) -> dict:
    """Send the payload to the FastAPI /predict endpoint."""
    return call_api("/predict", payload)


def call_explanation_api(payload: dict) -> dict:
    """
    Send the payload to the FastAPI /predict-with-explanation endpoint.
    Returns predicted_price, base_value, and shap_values dict.
    """
    return call_api("/predict-with-explanation", payload)


def draw_shap_chart(shap_values: dict, top_n: int = 8) -> plt.Figure:
    """
    Horizontal bar chart of the top N SHAP feature contributions.
    Green bars = pushed price UP, red bars = pushed price DOWN.

    KEY FIX vs the previous version:
    Value labels are now placed using `ax.annotate(..., textcoords="offset points")`
    instead of a data-unit offset (`pad = x_range * 0.03`). A data-unit offset only
    "looks right" when all bars are similar in magnitude — as soon as one feature's
    contribution dwarfs the others, that same percentage becomes a tiny pixel gap
    for the small bars (causing labels to sit on top of bars / collide with the
    zero-line / overlap each other) and a huge, wasted gap for the big bar.
    An offset in *points* is a constant number of pixels regardless of the data
    range, so labels never collide with their own bar or with each other.

    We also give the y-axis extra head/foot-room and let matplotlib's
    `constrained_layout` manage spacing instead of a hand-tuned
    `subplots_adjust`, which was fragile as the number of bars / feature-name
    lengths changed.
    """
    # Sort by absolute value descending, take top N, then reverse so the
    # most impactful feature renders at the top of the chart (barh is
    # bottom-to-top by default, so we feed items in ascending order).
    sorted_items = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    top_items    = list(reversed(sorted_items[:top_n]))

    # Replace underscores with spaces so "max_power" reads "max power"
    features = [item[0].replace("_", " ") for item in top_items]
    values   = [item[1] for item in top_items]
    n        = len(features)
    colors   = ["#16a34a" if v >= 0 else "#dc2626" for v in values]

    # --- Dynamic figure size ---
    row_height  = 0.75
    fig_height  = max(4.5, n * row_height + 1.4)
    fig_width   = 9.0
    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height),
        constrained_layout=True,  # lets matplotlib manage margins automatically
    )
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # Integer y-positions instead of string labels avoids matplotlib's
    # automatic label-to-bar mapping, which can cause vertical overlap.
    bars = ax.barh(range(n), values, color=colors, height=0.55, zorder=2)

    # --- X-axis limits: generous, symmetric-ish padding around the data ---
    x_min = min(values + [0])
    x_max = max(values + [0])
    span  = (x_max - x_min) or max(abs(x_max), 1)
    margin = span * 0.28
    ax.set_xlim(x_min - margin, x_max + margin)

    # --- Value labels: fixed pixel offset from the bar tip, so they never
    #     collide with the bar itself or with neighbouring labels, no matter
    #     how the bar magnitudes compare to each other. ---
    for bar, val in zip(bars, values):
        width = bar.get_width()
        if val >= 0:
            label, ha, dx = f"+₹{val:,.0f}", "left", 6
        else:
            label, ha, dx = f"₹{val:,.0f}", "right", -6
        ax.annotate(
            label,
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(dx, 0),
            textcoords="offset points",
            va="center", ha=ha,
            fontsize=8.5, color="white", fontweight="500",
            zorder=3, clip_on=False,
        )

    # --- Y-axis: feature name tick labels, with head/foot-room so the
    #     first and last bars' labels never bump into the axes edge. ---
    ax.set_yticks(range(n))
    ax.set_yticklabels(features, fontsize=9.5, color="#d1d5db")
    ax.set_ylim(-0.7, n - 0.3)

    ax.axvline(0, color="#6b7280", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlabel("Contribution to Price (₹)", color="#9ca3af", fontsize=9)
    ax.tick_params(axis="x", colors="#6b7280", labelsize=8)
    ax.tick_params(axis="y", length=0)        # hide y-axis tick marks (cosmetic)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#374151")
    ax.grid(axis="x", color="#1f2937", linewidth=0.6, zorder=0)

    return fig



# --------------------------------------------------------------------------- #
# HEADER / TITLE / DESCRIPTION
# --------------------------------------------------------------------------- #
st.markdown("<h1 style='text-align:center; margin-bottom:0.2rem;'>🚗 Car Price Predictor</h1>",
            unsafe_allow_html=True)
st.markdown(
    "<div class='app-subtitle'>"
    "Get an instant, AI-powered resale price estimate for a used car. "
    "Fill in the vehicle details below, then predict the price or see "
    "exactly which factors are driving the estimate."
    "</div>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# SECTION: CAR DETAILS
# --------------------------------------------------------------------------- #
st.subheader("📋 Car Details")

with st.container(border=True):
    # Row 1 — Age and Kilometres
    col1, col2 = st.columns(2, gap="large")
    with col1:
        vehicle_age = st.number_input(
            "📅 Vehicle Age (years)",
            min_value=0, max_value=50,
            value=5, step=1,
            help="How old is the car? e.g. 5 for a 2019 car bought in 2024",
        )
    with col2:
        km_driven = st.number_input(
            "🛣️ Kilometres Driven",
            min_value=0, max_value=1_000_000,
            value=40000, step=1000,
            help="Total distance driven since manufacture",
        )

    # Row 2 — Mileage and Engine
    col3, col4 = st.columns(2, gap="large")
    with col3:
        mileage = st.number_input(
            "⛽ Mileage (km/l)",
            min_value=1.0, max_value=60.0,
            value=18.5, step=0.5, format="%.1f",
            help="Fuel efficiency in kilometres per litre",
        )
    with col4:
        engine = st.number_input(
            "🔧 Engine (CC)",
            min_value=100, max_value=10_000,
            value=1197, step=50,
            help="Engine displacement in cubic centimetres, e.g. 1197",
        )

    # Row 3 — Max Power and Seats
    col5, col6 = st.columns(2, gap="large")
    with col5:
        max_power = st.number_input(
            "⚡ Max Power (bhp)",
            min_value=10.0, max_value=1000.0,
            value=82.0, step=1.0, format="%.1f",
            help="Maximum engine power in brake horsepower",
        )
    with col6:
        seats = st.number_input(
            "💺 Seats",
            min_value=2, max_value=10,
            value=5, step=1,
            help="Number of seats in the car",
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Row 4 — Dropdowns
    col7, col8, col9 = st.columns(3, gap="medium")
    with col7:
        fuel_type = st.selectbox(
            "⛽ Fuel Type",
            options=["Petrol", "Diesel", "CNG", "LPG", "Electric"],
            index=0,
        )
    with col8:
        seller_type = st.selectbox(
            "🏪 Seller Type",
            options=["Dealer", "Individual", "Trustmark Dealer"],
            index=0,
        )
    with col9:
        transmission_type = st.selectbox(
            "⚙️ Transmission",
            options=["Manual", "Automatic"],
            index=0,
        )

# --------------------------------------------------------------------------- #
# SECTION: ACTIONS
# --------------------------------------------------------------------------- #
st.subheader("🚀 Get Your Estimate")

btn_col1, btn_col2 = st.columns(2, gap="medium")
with btn_col1:
    predict_clicked = st.button("🔍 Predict Price", type="primary")
with btn_col2:
    explain_clicked = st.button("🧠 Explain Prediction", type="secondary")


# --------------------------------------------------------------------------- #
# SHARED PAYLOAD
# --------------------------------------------------------------------------- #
payload = {
    "vehicle_age":       int(vehicle_age),
    "km_driven":         int(km_driven),
    "mileage":           float(mileage),
    "engine":            int(engine),
    "max_power":         float(max_power),
    "seats":             int(seats),
    "fuel_type":         fuel_type,
    "seller_type":       seller_type,
    "transmission_type": transmission_type,
}


# --------------------------------------------------------------------------- #
# PREDICT  (existing behaviour, unchanged)
# --------------------------------------------------------------------------- #
def _show_api_error(e: Exception) -> None:
    """Centralised error display used by both buttons."""
    if isinstance(e, requests.exceptions.ConnectionError):
        st.error(
            f"❌ Could not connect to the prediction API at {BASE_URL}.\n\n"
            "Make sure the FastAPI backend is running:\n"
            "```\nuvicorn app:app --reload\n```"
        )
    elif isinstance(e, requests.exceptions.Timeout):
        st.error(f"❌ The API did not respond within {API_TIMEOUT} seconds. Please try again.")
    elif isinstance(e, requests.exceptions.HTTPError):
        status = e.response.status_code if e.response is not None else "unknown"
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"❌ API returned an error (HTTP {status}):\n\n{detail}")
    else:
        st.error(f"❌ Unexpected error: {e}")


if predict_clicked:
    with st.spinner("Calculating price estimate..."):
        try:
            result = call_predict_api(payload)
            price  = result["predicted_price"]

            st.subheader("💰 Prediction Result")

            metric_col, range_col1, range_col2 = st.columns(3, gap="medium")
            with metric_col:
                st.metric(
                    label="Estimated Selling Price",
                    value=format_inr(price),
                )
            with range_col1:
                st.metric(
                    label="Likely Low",
                    value=format_inr(price * 0.88),
                )
            with range_col2:
                st.metric(
                    label="Likely High",
                    value=format_inr(price * 1.12),
                )

            st.caption("Estimate based on current market data. "
                       "Actual selling price may vary.")
        except Exception as e:
            _show_api_error(e)


# --------------------------------------------------------------------------- #
# EXPLAIN  (new)
# --------------------------------------------------------------------------- #
if explain_clicked:
    with st.spinner("Running SHAP explanation..."):
        try:
            result = call_explanation_api(payload)
            price      = result["predicted_price"]
            base_value = result["base_value"]
            shap_vals  = result["shap_values"]

            st.subheader("💰 Prediction Result")

            price_col, base_col = st.columns(2, gap="medium")
            with price_col:
                st.metric(
                    label="Estimated Selling Price",
                    value=format_inr(price),
                    delta=format_inr(price - base_value),
                    help="Delta shows the total shift from the model's baseline prediction.",
                )
            with base_col:
                st.metric(
                    label="Baseline (Average Prediction)",
                    value=format_inr(base_value),
                )

            st.write("")

            # --- Top contributors summary ---
            sorted_shap = sorted(shap_vals.items(), key=lambda x: x[1], reverse=True)
            positive    = [(f, v) for f, v in sorted_shap if v > 0]
            negative    = [(f, v) for f, v in sorted_shap if v < 0]

            st.subheader("📊 What's Driving This Price?")

            with st.container(border=True):
                st.caption(
                    f"Starting from a baseline of **{format_inr(base_value)}** "
                    f"(model's average prediction), these features pushed the "
                    f"estimate to **{format_inr(price)}**."
                )

                col_pos, col_neg = st.columns(2, gap="large")

                with col_pos:
                    st.markdown("**🟢 Pushed price UP**")
                    for feat, val in positive[:5]:
                        label = feat.replace("_", " ").title()
                        st.markdown(f"- `{label}` &nbsp; **+{format_inr(val)}**",
                                    unsafe_allow_html=True)

                with col_neg:
                    st.markdown("**🔴 Pushed price DOWN**")
                    for feat, val in negative[:5]:
                        label = feat.replace("_", " ").title()
                        st.markdown(f"- `{label}` &nbsp; **{format_inr(val)}**",
                                    unsafe_allow_html=True)

            # --- SHAP bar chart ---
            st.write("")
            st.subheader("🔬 Feature Contribution Chart")
            st.caption("Top 8 features by absolute SHAP contribution. "
                       "Green = raised the price, Red = lowered the price.")
            fig = draw_shap_chart(shap_vals, top_n=8)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        except Exception as e:
            _show_api_error(e)

# --------------------------------------------------------------------------- #
# FOOTER
# --------------------------------------------------------------------------- #
st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.caption(
    "⚠️ Predictions are estimates based on historical data "
    "and may not reflect actual market prices. "
    "This tool is intended for informational purposes only."
)