import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClimaKrishiTrace — Rice Impact Predictor",
    page_icon="🌾",
    layout="centered"
)

# ── Load Models ────────────────────────────────────────────────────────────────
model_conv    = joblib.load("model_conventional.pkl")
scaler_conv   = joblib.load("scaler_conventional.pkl")
model_org     = joblib.load("model_organic.pkl")
scaler_org    = joblib.load("scaler_organic.pkl")

# ── Constants ──────────────────────────────────────────────────────────────────
CONV_RANGES = {'N': (120, 150), 'P': (40, 60), 'K': (30, 40), 'Zn': (10, 30)}
CONV_DEFAULTS = {'N': 135.0, 'P': 50.0, 'K': 35.0, 'Zn': 20.0}
ORG_DEFAULTS  = {'Manure': 10000.0, 'Compost': 1500.0}

# Indian market rates (₹/kg)
COST_RATES = {
    'N_synthetic':  6.5,    # ₹/kg N (Urea basis)
    'P_synthetic':  28.0,   # ₹/kg P (DAP basis)
    'K_synthetic':  15.0,   # ₹/kg K (MOP basis)
    'Zn_synthetic': 120.0,  # ₹/kg Zn
    'Manure':       0.8,    # ₹/kg FYM
    'Compost':      3.5,    # ₹/kg Compost
}

IMPACT_LABELS = ["🌍 Global Warming", "💧 Freshwater Eutrophication", "🌫️ Terrestrial Acidification", "☠️ Terrestrial Ecotoxicity"]
IMPACT_UNITS  = ["kg CO₂-eq", "kg P-eq", "kg SO₂-eq", "CTUe"]
IMPACT_FORMATS = ["{:,.2f}", "{:.6f}", "{:.4f}", "{:,.2f}"]

# ── Prediction Functions ───────────────────────────────────────────────────────
def predict_conventional(N, P, K, Zn):
    inputs = pd.DataFrame([[N, P, K, Zn]], columns=['N_rate', 'P_rate', 'K_rate', 'Zn_rate'])
    scaled = scaler_conv.transform(inputs)
    return model_conv.predict(scaled)[0]

def predict_organic(manure, compost):
    inputs = pd.DataFrame([[manure, compost]], columns=['Manure_rate', 'Compost_rate'])
    scaled = scaler_org.transform(inputs)
    return model_org.predict(scaled)[0]

def blend(conv_output, org_output, alpha):
    return (1 - alpha) * np.array(conv_output) + alpha * np.array(org_output)

def calc_cost(N, P, K, Zn, manure, compost, alpha):
    conv_cost = (N  * COST_RATES['N_synthetic'] +
                 P  * COST_RATES['P_synthetic'] +
                 K  * COST_RATES['K_synthetic'] +
                 Zn * COST_RATES['Zn_synthetic'])
    org_cost  = (manure  * COST_RATES['Manure'] +
                 compost * COST_RATES['Compost'])
    return (1 - alpha) * conv_cost + alpha * org_cost

def validate_conv(N, P, K, Zn):
    vals = {'N': N, 'P': P, 'K': K, 'Zn': Zn}
    return [
        f"**{k}** = {v} kg/ha (valid: {CONV_RANGES[k][0]}–{CONV_RANGES[k][1]} kg/ha)"
        for k, v in vals.items()
        if not (CONV_RANGES[k][0] <= v <= CONV_RANGES[k][1])
    ]

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("🌾 ClimaKrishiTrace")
st.markdown(
    "Predict and compare the **full environmental impact** of conventional vs organic "
    "fertiliser management in rice cultivation — powered by ISO 14040/44-compliant LCA "
    "and machine learning."
)
st.markdown("---")

# ── Tab Layout ─────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎚️ Organic–Conventional Gradient", "🔬 Single System Predictor"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — GRADIENT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    st.subheader("Organic–Conventional Transition Analyser")
    st.markdown(
        "Enter your conventional and organic input rates, then use the slider to explore "
        "the environmental and economic impact at any point along the transition."
    )
    st.markdown("---")

    # ── Inputs ────────────────────────────────────────────────────────────────
    col_conv, col_org = st.columns(2)

    with col_conv:
        st.markdown("### 🧪 Conventional Inputs (kg/ha)")
        N  = st.number_input("Nitrogen (N)",   min_value=0.0, value=135.0, step=0.5, key="gN")
        P  = st.number_input("Phosphorus (P)", min_value=0.0, value=50.0,  step=0.5, key="gP")
        K  = st.number_input("Potassium (K)",  min_value=0.0, value=35.0,  step=0.5, key="gK")
        Zn = st.number_input("Zinc (Zn)",      min_value=0.0, value=20.0,  step=0.5, key="gZn")
        st.caption("Valid ranges — N: 120–150 | P: 40–60 | K: 30–40 | Zn: 10–30")

    with col_org:
        st.markdown("### 🌿 Organic Inputs (kg/ha)")
        manure  = st.number_input("Farm Yard Manure",  min_value=0.0, value=10000.0, step=100.0, key="gManure")
        compost = st.number_input("Compost",           min_value=0.0, value=1500.0,  step=50.0,  key="gCompost")
        st.caption("Valid ranges — Manure: 5,000–15,000 | Compost: 1,000–2,000 kg/ha")

    warnings = validate_conv(N, P, K, Zn)
    if warnings:
        st.warning("⚠️ Conventional inputs out of range:\n\n" + "\n".join(f"- {w}" for w in warnings))

    st.markdown("---")

    # ── Slider ────────────────────────────────────────────────────────────────
    st.subheader("🎚️ Organic Input Fraction")
    alpha = st.slider(
        "0% = Fully Conventional    →    100% = Fully Organic",
        min_value=0, max_value=100, value=0, step=5
    ) / 100

    label_left  = "🧪 Fully Conventional" if alpha == 0.0 else ""
    label_right = "🌿 Fully Organic"       if alpha == 1.0 else ""
    if label_left:
        st.info(f"Currently showing: **{label_left}**")
    elif label_right:
        st.success(f"Currently showing: **{label_right}**")
    else:
        st.info(f"Currently showing: **{int(alpha*100)}% Organic / {int((1-alpha)*100)}% Conventional blend**")

    st.markdown("---")

    # ── Calculate ─────────────────────────────────────────────────────────────
    conv_out  = predict_conventional(N, P, K, Zn)
    org_out   = predict_organic(manure, compost)
    blend_out = blend(conv_out, org_out, alpha)

    conv_cost  = calc_cost(N, P, K, Zn, manure, compost, 0.0)
    org_cost   = calc_cost(N, P, K, Zn, manure, compost, 1.0)
    blend_cost = calc_cost(N, P, K, Zn, manure, compost, alpha)

    # ── Impact Results ────────────────────────────────────────────────────────
    st.subheader("📊 Environmental Impact at Selected Blend")
    st.caption("Includes upstream fertiliser production + field-level emissions (N₂O, NO₃, NH₃, PO₄)")

    for i, (label, unit, fmt) in enumerate(zip(IMPACT_LABELS, IMPACT_UNITS, IMPACT_FORMATS)):
        col1, col2, col3 = st.columns(3)
        delta_pct = ((blend_out[i] - conv_out[i]) / conv_out[i]) * 100
        with col1:
            st.metric(f"{label} — Conv.", f"{fmt.format(conv_out[i])} {unit}")
        with col2:
            st.metric(f"{label} — Blend", f"{fmt.format(blend_out[i])} {unit}",
                      delta=f"{delta_pct:+.1f}% vs Conv.")
        with col3:
            st.metric(f"{label} — Organic", f"{fmt.format(org_out[i])} {unit}")

    st.markdown("---")

    # ── Cost Layer ────────────────────────────────────────────────────────────
    st.subheader("💰 Input Cost Analysis (₹/ha)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🧪 Conventional Cost", f"₹{conv_cost:,.0f}/ha")
    with col2:
        cost_delta = blend_cost - conv_cost
        st.metric("🎚️ Blend Cost", f"₹{blend_cost:,.0f}/ha",
                  delta=f"₹{cost_delta:+,.0f} vs Conv.")
    with col3:
        st.metric("🌿 Organic Cost", f"₹{org_cost:,.0f}/ha")

    # Cost per kg CO2 avoided
    gwp_reduction = conv_out[0] - blend_out[0]
    if gwp_reduction > 0:
        cost_per_co2 = cost_delta / gwp_reduction
        st.info(f"💡 At this blend — reducing **{gwp_reduction:.1f} kg CO₂-eq/ha** costs an additional **₹{cost_per_co2:,.1f} per kg CO₂ avoided**" if cost_delta > 0
                else f"💡 At this blend — reducing **{gwp_reduction:.1f} kg CO₂-eq/ha** while **saving ₹{abs(cost_delta):,.0f}/ha**")
    elif gwp_reduction < 0:
        st.warning("⚠️ This blend increases GWP compared to conventional — consider adjusting your organic inputs.")
    else:
        st.info("No GWP change at this blend point.")

    st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# 🌱 CARBON CREDIT CALCULATOR (CCTS — 1 ha basis)
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("🌱 Carbon Credit Potential (CCTS)")
st.caption("All results are calculated per hectare (1 ha) — consistent with LCA functional unit")

# ── Fixed Functional Unit ─────────────────────────────────────────────────────
field_size = 1.0

# ── Inputs ────────────────────────────────────────────────────────────────────
buffer_pct = st.slider(
    "Permanence Buffer (%)",
    min_value=10, max_value=30, value=20, step=1,
    help="Applied only to soil carbon credits"
)

# ── SOC Scenario ──────────────────────────────────────────────────────────────
soc_choice = st.selectbox(
    "Soil Carbon Scenario",
    [
        "Low Sequestration (Baseline Systems)",
        "Typical Sequestration (Improved Practices)",
        "High Sequestration (Advanced Systems)"
    ]
)

if soc_choice == "Low Sequestration (Baseline Systems)":
    SOC_RATE = 0.4
elif soc_choice == "Typical Sequestration (Improved Practices)":
    SOC_RATE = 0.8
else:
    SOC_RATE = 1.155

# ── Constants ─────────────────────────────────────────────────────────────────
C_TO_CO2 = 3.667
CCTS_LOW  = 600
CCTS_HIGH = 900

buffer_fraction = buffer_pct / 100

# ── Credits Calculation ───────────────────────────────────────────────────────
# Emission Reduction Credits (per ha)
er_credits = max(0, (conv_out[0] - blend_out[0]) / 1000)

# Soil Carbon Credits (per ha)
soc_gross = SOC_RATE * C_TO_CO2
soc_credits = soc_gross * (1 - buffer_fraction)

# Total (per ha)
total_credits = er_credits + soc_credits

# ── Market Value (per ha) ─────────────────────────────────────────────────────
ccts_low_val  = total_credits * CCTS_LOW
ccts_high_val = total_credits * CCTS_HIGH

# ── Display ───────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📉 Emission Reduction (per ha)", f"{er_credits:.3f} t CO₂-eq")

with col2:
    st.metric("🌱 Soil Carbon (per ha)", f"{soc_credits:.3f} t CO₂-eq")

with col3:
    st.metric("🏆 Total Credits (per ha)", f"{total_credits:.3f} t CO₂-eq")

st.markdown("##### 💰 Estimated Value (per ha, CCTS)")

st.info(
    f"₹{ccts_low_val:,.0f} – ₹{ccts_high_val:,.0f}\n\n"
    f"₹{CCTS_LOW}–₹{CCTS_HIGH} per t CO₂-eq"
)
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SINGLE SYSTEM PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    st.subheader("Single System Impact Predictor")
    system = st.radio("Select System", ["🧪 Conventional", "🌿 Organic"], horizontal=True)
    st.markdown("---")

    if system == "🧪 Conventional":

        st.subheader("Enter Fertiliser Application Rates (kg/ha)")
        st.info("**Recommended Ranges:** N: 120–150 | P: 40–60 | K: 30–40 | Zn: 10–30")

        col1, col2 = st.columns(2)
        with col1:
            sN  = st.number_input("Nitrogen (N)",   min_value=0.0, value=135.0, step=0.5, key="sN")
            sK  = st.number_input("Potassium (K)",  min_value=0.0, value=35.0,  step=0.5, key="sK")
        with col2:
            sP  = st.number_input("Phosphorus (P)", min_value=0.0, value=50.0,  step=0.5, key="sP")
            sZn = st.number_input("Zinc (Zn)",      min_value=0.0, value=20.0,  step=0.5, key="sZn")

        warn = validate_conv(sN, sP, sK, sZn)
        if warn:
            st.error("🚨 Out of range:\n\n" + "\n".join(f"- {w}" for w in warn))
        else:
            st.success("✅ All inputs within recommended ranges.")

        if st.button("🔍 Predict", use_container_width=True, key="btn_conv"):
            out = predict_conventional(sN, sP, sK, sZn)
            st.subheader("📊 Predicted Environmental Impact")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🌍 Global Warming",            f"{out[0]:,.2f} kg CO₂-eq")
                st.metric("💧 Freshwater Eutrophication", f"{out[1]:.6f} kg P-eq")
            with col2:
                st.metric("🌫️ Terrestrial Acidification", f"{out[2]:.4f} kg SO₂-eq")
                st.metric("☠️ Terrestrial Ecotoxicity",   f"{out[3]:,.2f} CTUe")

    else:

        st.subheader("Enter Organic Amendment Rates (kg/ha)")
        st.info("**Recommended Ranges:** Manure: 5,000–15,000 | Compost: 1,000–2,000 kg/ha")

        col1, col2 = st.columns(2)
        with col1:
            sManure  = st.number_input("Farm Yard Manure (kg/ha)", min_value=0.0, value=10000.0, step=100.0, key="sManure")
        with col2:
            sCompost = st.number_input("Compost (kg/ha)",          min_value=0.0, value=1500.0,  step=50.0,  key="sCompost")

        if st.button("🔍 Predict", use_container_width=True, key="btn_org"):
            out = predict_organic(sManure, sCompost)
            st.subheader("📊 Predicted Environmental Impact")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🌍 Global Warming",            f"{out[0]:,.2f} kg CO₂-eq")
                st.metric("💧 Freshwater Eutrophication", f"{out[1]:.6f} kg P-eq")
            with col2:
                st.metric("🌫️ Terrestrial Acidification", f"{out[2]:.4f} kg SO₂-eq")
                st.metric("☠️ Terrestrial Ecotoxicity",   f"{out[3]:,.2f} CTUe")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ About ClimaKrishiTrace"):
    st.markdown("""
    **Algorithm:** Ridge Regression (Multivariate, L2-regularised) — trained separately for
    conventional and organic rice systems.

    **System Boundary:** Cradle-to-field — upstream fertiliser/amendment production, transport,
    and field-level emissions (N₂O, NO₃⁻, NH₃, PO₄³⁻) via IPCC and SALCA methodologies.

    **Training Data:** 1,000 LCA simulations (conventional) + 600 LCA simulations (organic)
    run in OpenLCA using the ecoinvent database with parametric sampling.

    **Primary Data:** ICAR-Indian Institute of Rice Research (IIRR), Hyderabad
    and Krishi Vigyan Kendra (KVK), Medak, Telangana.

    **Impact Assessment:** ReCiPe 2016 Midpoint (H)

    **Cost References:** Indian market rates for fertilisers and organic amendments
    (government and institutional datasets).
    """)