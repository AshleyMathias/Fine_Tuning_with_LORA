# app.py

import streamlit as st
from fine_tuning2 import prepare_dataset, tokenize_dataset, build_lora_model, train_model, predict_sentiment

st.set_page_config(page_title="Train BERT with LoRA", layout="centered")
st.title("🤖 Train Your Own BERT Model with LoRA")
st.markdown("Enter your sentences and train a mini BERT classifier with LoRA adapters, right in your browser!")

# --- Step 1: Input Sentences ---
st.subheader("📥 Step 1: Input Sentences & Labels")

num_inputs = st.slider("How many training examples?", min_value=2, max_value=10, value=4)

# Optional: Autofill examples
if st.button("🔁 Use Example Sentences"):
    st.session_state.auto_texts = [
        "I absolutely loved this movie!",
        "The performance was brilliant.",
        "What a fantastic storyline!",
        "Truly an amazing experience.",
        "This movie was a complete disaster.",
        "I hated the plot and the acting.",
        "It was painfully boring to watch.",
        "Worst film I have ever seen."
    ]
    st.session_state.auto_labels = ["Positive"] * 4 + ["Negative"] * 4
    num_inputs = 8

texts, labels = [], []
label_map = {"Positive": 1, "Negative": 0}

for i in range(num_inputs):
    col1, col2 = st.columns([3, 1])
    with col1:
        text = st.text_input(f"Sentence {i+1}", key=f"text_{i}", value=st.session_state.get("auto_texts", [""]*num_inputs)[i] if "auto_texts" in st.session_state else "")
    with col2:
        label = st.selectbox(
            f"Label {i+1}",
            ["Positive", "Negative"],
            key=f"label_{i}",
            index=(0 if "auto_labels" not in st.session_state else (0 if st.session_state["auto_labels"][i] == "Positive" else 1))
        )
    texts.append(text)
    labels.append(label_map[label])

if num_inputs < 6:
    st.info("ℹ️ More training examples (6+) are recommended for better accuracy.")

train_button = st.button("✅ Train BERT with LoRA")

# --- Step 2: Training ---
if train_button:
    if any(t.strip() == "" for t in texts):
        st.warning("⚠️ Please fill in all sentences.")
    else:
        with st.spinner("Preparing dataset..."):
            dataset = prepare_dataset(texts, labels)
            model, tokenizer = build_lora_model()
            tokenized = tokenize_dataset(dataset, tokenizer)

        st.success("✅ Dataset ready!")

        with st.spinner("Training the model (LoRA on CPU)..."):
            model = train_model(model, tokenizer, tokenized)
        st.success("🎉 Model trained successfully!")

        st.session_state.model = model
        st.session_state.tokenizer = tokenizer

# --- Step 3: Inference ---
if "model" in st.session_state:
    st.subheader("🔍 Step 3: Predict Sentiment on New Text")
    new_text = st.text_input("Enter a sentence for prediction:")
    if st.button("🚀 Predict"):
        sentiment, confidence, probs = predict_sentiment(
            st.session_state.model,
            st.session_state.tokenizer,
            new_text,
            return_probs=True  # you'll need to support this in the function
        )
        st.write(f"**Predicted Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.2f}")
        st.write(f"**Class Probabilities:** Positive = {probs[1]:.2f}, Negative = {probs[0]:.2f}")
