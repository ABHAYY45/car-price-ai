"""
streamlit_app.py
----------------
Streamlit frontend for the Car Price Prediction FastAPI backend.

Run from the project root:
    streamlit run streamlit_app.py

The FastAPI backend must be running separately on port 8000:
    uvicorn app:app --reload
"""

import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
API_URL     = "http://127.0.0.1:8000/predict"
API_TIMEOUT = 10  # seconds before giving up on the request

# --------------------------------------------------------------------------- #
# PAGE SETUP
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
)

# --------------------------------------------------------------------------- #
# CUSTOM CSS
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
    .block-container { max-width: 860px; padding-top: 2rem; padding-bottom: 3rem; }

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


def call_predict_api(payload: dict) -> dict:
    """
    Send the payload to the FastAPI /predict endpoint.
    Raises requests.RequestException on any network or HTTP error.
    """
    response = requests.post(API_URL, json=payload, timeout=API_TIMEOUT)
    response.raise_for_status()  # raises HTTPError for 4xx / 5xx responses
    return response.json()


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.markdown("<h1 style='text-align:center;'>🚗 Car Price Predictor</h1>",
            unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Enter the car details below to get an "
            "instant AI-powered price estimate</div>",
            unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# INPUT FORM
# --------------------------------------------------------------------------- #
with st.container(border=True):
    st.markdown("#### 📋 Car Details")

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
# PREDICT BUTTON + RESULT
# --------------------------------------------------------------------------- #
predict_clicked = st.button("🔍 Predict Price", type="primary")

if predict_clicked:
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

    with st.spinner("Calculating price estimate..."):
        try:
            result = call_predict_api(payload)
            price  = result["predicted_price"]

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Estimated Selling Price</div>
                <p class="result-price">{format_inr(price)}</p>
                <div class="result-note">
                    Based on current market data · Likely range:
                    {format_inr(price * 0.88)} – {format_inr(price * 1.12)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Could not connect to the prediction API.\n\n"
                "Make sure the FastAPI backend is running:\n"
                "```\nuvicorn app:app --reload\n```"
            )
        except requests.exceptions.Timeout:
            st.error(
                f"❌ The API did not respond within {API_TIMEOUT} seconds. "
                "Please try again."
            )
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            try:
                detail = e.response.json().get("detail", str(e))
            except Exception:
                detail = str(e)
            st.error(f"❌ API returned an error (HTTP {status}):\n\n{detail}")
        except (KeyError, ValueError):
            st.error(
                "❌ Unexpected response from API. "
                "The response did not contain 'predicted_price'."
            )
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

# --------------------------------------------------------------------------- #
# FOOTER
# --------------------------------------------------------------------------- #
st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "⚠️ Predictions are estimates based on historical data "
    "and may not reflect actual market prices."
)