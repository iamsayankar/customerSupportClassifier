import streamlit as st

from predict import predict_ticket

st.set_page_config(
    page_title="Customer Support Classifier",
    page_icon="🎫",
    layout="centered",
)

st.title("🎫 Customer Support Ticket Classifier")
st.write("Enter a customer support ticket description and predict its category.")

description = st.text_area(
    "Ticket Description",
    placeholder="Example: I forgot my password and cannot login.",
)

if st.button("Predict Category", type="primary"):
    if not description.strip():
        st.warning("Please enter a ticket description.")
    else:
        try:
            category, confidence = predict_ticket(description)
            st.success(f"Predicted Category: **{category}**")
            st.info(f"Confidence: **{confidence:.2f}%**")
        except ValueError as exc:
            st.warning(str(exc))

st.markdown("---")
st.caption("AI Fresher Technical Assignment | TF-IDF + Logistic Regression")