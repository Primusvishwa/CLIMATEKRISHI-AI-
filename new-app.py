import streamlit as st
import pandas as pd
import joblib

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rice Fertiliser Impact Predictor",
    page_icon="🌱",
    layout="centered"
)

# ── Load Model + Scaler ────────────────────────────────────────────────────────
model  = joblib.load("model_conventional.pkl")
scaler = joblib.load("scaler_conventional.pkl")

# ── Helpers ────────────────────────────────────────────────────────────────────
RANGES = {'N': (120, 150), 'P': (40, 60), 'K': (30, 40), 'Zn': (10, 30)}

def predict(N, P, K, Zn):
    inputs = pd.DataFrame([[N, P, K, Zn]], columns=['N_rate', 'P_rate', 'K_rate', 'Zn_rate'])
    scaled = scaler.transform(inputs)
    pred   = model.predict(scaled)[0]
    return pred[0], pred[1], pred[2], pred[3]

def validate(N, P, K, Zn):
    vals = {'N': N, 'P': P, 'K': K, 'Zn': Zn}
    return [
        f"**{k}** = {v} kg/ha (valid range: {RANGES[k][0]}–{RANGES[k][1]} kg/ha)"
        for k, v in vals.items()
        if not (RANGES[k][0] <= v <= RANGES[k][1])
    ]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🌱 Rice Fertiliser Impact Predictor")
st.markdown(
    "Predict the **full environmental impact** of fertiliser application in rice cultivation — "
    "including upstream production emissions and field-level emissions, without needing a full Life Cycle Assessment."
)
st.markdown("---")

# ── Mode Toggle ────────────────────────────────────────────────────────────────
mode = st.radio("Select Mode", ["Single Prediction", "Compare Two Combinations"], horizontal=True)
st.markdown("---")
st.info("**Recommended Ranges (kg/ha):** N: 120–150  |  P: 40–60  |  K: 30–40  |  Zn: 10–30\n\n⚠️ Values outside these ranges may give unreliable predictions.")

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Single Prediction":

    st.subheader("Enter Fertiliser Application Rates (kg/ha)")
    col1, col2 = st.columns(2)
    with col1:
        N  = st.number_input("Nitrogen (N)",   min_value=0.0, value=135.0, step=0.5)
        K  = st.number_input("Potassium (K)",  min_value=0.0, value=35.0,  step=0.5)
    with col2:
        P  = st.number_input("Phosphorus (P)", min_value=0.0, value=50.0,  step=0.5)
        Zn = st.number_input("Zinc (Zn)",      min_value=0.0, value=20.0,  step=0.5)

    warnings = validate(N, P, K, Zn)
    if warnings:
        st.error("🚨 Out of range:\n\n" + "\n".join(f"- {w}" for w in warnings))
    else:
        st.success("✅ All inputs within recommended ranges.")

    st.markdown("---")

    if st.button("🔍 Predict Environmental Impact", use_container_width=True):
        gwp, eu, ac, eco = predict(N, P, K, Zn)

        st.subheader("📊 Predicted Environmental Impact Scores")
        st.caption("Includes upstream fertiliser production + field-level emissions (N₂O, NO₃, NH₃, PO₄)")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🌍 Global Warming",            f"{gwp:,.2f} kg CO₂-eq")
            st.metric("💧 Freshwater Eutrophication", f"{eu:.6f} kg P-eq")
        with col2:
            st.metric("🌫️ Terrestrial Acidification", f"{ac:.4f} kg SO₂-eq")
            st.metric("☠️ Terrestrial Ecotoxicity",   f"{eco:,.2f} CTUe")

        st.markdown("---")
        st.subheader("📖 What do these mean?")

        with st.expander("🌍 Global Warming (GWP)"):
            st.markdown("""
            **Global Warming Potential (GWP)** measures how much a process contributes to climate change,
            expressed in **kg CO₂-equivalent**. In rice farming, GWP captures both the energy-intensive
            production of synthetic fertilisers and direct field emissions — N₂O (273× more potent than CO₂)
            and CH₄ from flooded paddy fields (27.9× more potent than CO₂).
            """)

        with st.expander("💧 Freshwater Eutrophication"):
            st.markdown("""
            **Freshwater Eutrophication** measures nutrient enrichment in freshwater bodies,
            expressed in **kg Phosphorus-equivalent**. Excess phosphorus and nitrate from fertiliser
            application leach into water bodies, triggering algal blooms that deplete oxygen and
            devastate aquatic ecosystems.
            """)

        with st.expander("🌫️ Terrestrial Acidification"):
            st.markdown("""
            **Terrestrial Acidification** measures soil and ecosystem acidification,
            expressed in **kg SO₂-equivalent**. Ammonia (NH₃) released from nitrogen fertilisers
            is the primary driver — it deposits onto soil and vegetation, lowering pH and
            reducing biodiversity over time.
            """)

        with st.expander("☠️ Terrestrial Ecotoxicity"):
            st.markdown("""
            **Terrestrial Ecotoxicity** measures the toxic impact on land-based ecosystems,
            expressed in **CTUe** (Comparative Toxic Units for ecosystems). Zinc is the dominant
            driver — excess Zn accumulates in soil and becomes toxic to soil organisms and plants
            at elevated concentrations.
            """)

        st.markdown("---")
        st.caption(
            "Model: Ridge Regression trained on ISO 14040/44-compliant LCA simulations (OpenLCA + ecoinvent). "
            "Primary data: ICAR-IIRR, Hyderabad and KVK, Medak. "
            "Results are predictions — not a substitute for full LCA."
        )

# ══════════════════════════════════════════════════════════════════════════════
# COMPARE MODE
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.subheader("Compare Two Fertiliser Combinations")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🅰️ Combination A")
        N_a  = st.number_input("Nitrogen (N) — A",   min_value=0.0, value=120.0, step=0.5, key="Na")
        P_a  = st.number_input("Phosphorus (P) — A", min_value=0.0, value=40.0,  step=0.5, key="Pa")
        K_a  = st.number_input("Potassium (K) — A",  min_value=0.0, value=30.0,  step=0.5, key="Ka")
        Zn_a = st.number_input("Zinc (Zn) — A",      min_value=0.0, value=10.0,  step=0.5, key="Zna")

    with col_b:
        st.markdown("### 🅱️ Combination B")
        N_b  = st.number_input("Nitrogen (N) — B",   min_value=0.0, value=150.0, step=0.5, key="Nb")
        P_b  = st.number_input("Phosphorus (P) — B", min_value=0.0, value=60.0,  step=0.5, key="Pb")
        K_b  = st.number_input("Potassium (K) — B",  min_value=0.0, value=40.0,  step=0.5, key="Kb")
        Zn_b = st.number_input("Zinc (Zn) — B",      min_value=0.0, value=30.0,  step=0.5, key="Znb")

    warn_a = validate(N_a, P_a, K_a, Zn_a)
    warn_b = validate(N_b, P_b, K_b, Zn_b)
    if warn_a:
        st.error("🚨 Combination A out of range:\n\n" + "\n".join(f"- {w}" for w in warn_a))
    if warn_b:
        st.error("🚨 Combination B out of range:\n\n" + "\n".join(f"- {w}" for w in warn_b))

    st.markdown("---")

    if st.button("🔍 Compare Combinations", use_container_width=True):
        gwp_a, eu_a, ac_a, eco_a = predict(N_a, P_a, K_a, Zn_a)
        gwp_b, eu_b, ac_b, eco_b = predict(N_b, P_b, K_b, Zn_b)

        st.subheader("📊 Comparison Results")
        st.caption("Includes upstream fertiliser production + field-level emissions (N₂O, NO₃, NH₃, PO₄)")

        categories = ["🌍 Global Warming", "💧 Freshwater Eutrophication", "🌫️ Terrestrial Acidification", "☠️ Terrestrial Ecotoxicity"]
        units      = ["kg CO₂-eq", "kg P-eq", "kg SO₂-eq", "CTUe"]
        values_a   = [gwp_a, eu_a, ac_a, eco_a]
        values_b   = [gwp_b, eu_b, ac_b, eco_b]
        formats    = ["{:,.2f}", "{:.6f}", "{:.4f}", "{:,.2f}"]

        for i, cat in enumerate(categories):
            col1, col2, col3 = st.columns([2, 2, 1])
            fmt    = formats[i]
            delta  = ((values_b[i] - values_a[i]) / values_a[i]) * 100
            winner = "🅰️ Lower" if values_a[i] < values_b[i] else "🅱️ Lower"
            with col1:
                st.metric(f"{cat} — A", f"{fmt.format(values_a[i])} {units[i]}")
            with col2:
                st.metric(f"{cat} — B", f"{fmt.format(values_b[i])} {units[i]}", delta=f"{delta:+.1f}% vs A")
            with col3:
                st.markdown(f"<br><b>{winner}</b>", unsafe_allow_html=True)

        st.markdown("---")
        st.caption("Lower values = less environmental impact. Δ% shown relative to Combination A.")

# ── Model Info ─────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ About the Model"):
    st.markdown("""
    **Algorithm:** Ridge Regression (Multivariate, L2-regularised)

    **Training Data:** 1000 LCA simulations run in OpenLCA using the ecoinvent database,
    with parametric sampling across agronomically valid NPK + Zn input ranges for conventional
    Indian rice cultivation.

    **System Boundary:** Cradle-to-field — includes upstream fertiliser production, transport,
    and field-level emissions (N₂O, NO₃⁻, NH₃, PO₄³⁻) calculated using IPCC and SALCA methodologies.

    **Primary Data Sources:** ICAR-Indian Institute of Rice Research (IIRR), Hyderabad
    and Krishi Vigyan Kendra (KVK), Medak, Telangana.

    **Impact Assessment Method:** ReCiPe 2016 Midpoint (H)
    """)