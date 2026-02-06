import streamlit as st
import pickle
import numpy as np

# Load trained model & encoder
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

st.set_page_config(
    page_title="Fake Discount Detector",
    layout="centered"
)

st.title("🛒 Fake Discount Detector (Amazon Products)")
st.caption("AI-powered system to detect misleading e-commerce discounts")

st.divider()

# -------------------------
# USER INPUTS
# -------------------------
product = st.text_input("📦 Product Name")
category = st.selectbox("📂 Product Category", encoder.classes_)

col1, col2 = st.columns(2)

with col1:
    original_price = st.number_input("Original Price (₹)", min_value=1.0)

with col2:
    discounted_price = st.number_input("Discounted Price (₹)", min_value=1.0)

avg_past_price = st.number_input(
    "Average Past Price (₹)",
    min_value=1.0,
    help="Historical average price for this category"
)

st.divider()

# -------------------------
# PREDICTION
# -------------------------
if st.button("🔍 Check Discount", use_container_width=True):

    if discounted_price > original_price:
        st.error("Discounted price cannot be higher than original price!")
        st.stop()

    # Feature engineering
    discount_percentage = ((original_price - discounted_price) / original_price) * 100
    price_inflation = original_price - avg_past_price
    discount_vs_history = avg_past_price - discounted_price
    category_encoded = encoder.transform([category])[0]

    X = np.array([[
        discount_percentage,
        price_inflation,
        discount_vs_history,
        category_encoded
    ]])

    # Prediction
    prob_fake = model.predict_proba(X)[0][1]

    st.subheader("📊 Prediction Result")

    # Probability bar
    st.progress(int(prob_fake * 100))
    st.write(f"**Fake Discount Probability:** `{prob_fake*100:.2f}%`")

    if prob_fake >= 0.6:
        st.error("❌ Likely a FAKE discount")
    elif prob_fake >= 0.4:
        st.warning("⚠️ Suspicious discount – verify before buying")
    else:
        st.success("✅ Likely a REAL discount")

    st.divider()

    # -------------------------
    # EXPLANATION SECTION
    # -------------------------
    st.subheader("🧠 Why this result?")

    st.write("**Key factors considered:**")
    st.markdown(
        f"""
        • Discount Percentage: **{discount_percentage:.2f}%**  
        • Price Inflation: **₹{price_inflation:.2f}**  
        • Discount vs Historical Price: **₹{discount_vs_history:.2f}**  
        • Product Category: **{category}**
        """
    )

    st.divider()

    # -------------------------
    # CONSUMER INSIGHTS
    # -------------------------
    st.subheader("💡 Smart Consumer Tips")

    if price_inflation > 0:
        st.warning("Original price is higher than historical category average.")

    if discounted_price > avg_past_price:
        st.warning("Even after discount, price is above historical average.")

    if discount_percentage > 50:
        st.info("Very high discounts are often marketing tactics.")

    if prob_fake >= 0.6:
        st.info("Compare prices across multiple platforms before purchasing.")

    st.caption("⚠ This prediction is based on historical category pricing patterns.")
