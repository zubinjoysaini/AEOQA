import streamlit as st
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("⚠️ OPENAI_API_KEY not found. Please set it in your .env file.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Auto Q&A Generator", page_icon="🤖", layout="centered")

st.title("🤖 Topic-based Question & Answer Generator")
st.markdown("Enter a topic, and this app will generate related questions and answers using ChatGPT.")

topic = st.text_input("Enter a topic:")
num_questions = st.slider("Number of questions to generate:", 3, 15, 5)

if st.button("Generate Q&A"):
    if not topic.strip():
        st.warning("Please enter a topic before generating.")
        st.stop()

    with st.spinner("🧠 Generating questions..."):
        q_prompt = f"Generate {num_questions} natural, varied questions that people might ask about '{topic}'."
        q_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": q_prompt}]
        )
        questions = [q.strip("- ").strip() for q in q_response.choices[0].message.content.split("\n") if q.strip()]

    qa_data = []
    st.subheader("📋 Generated Q&A")
    progress = st.progress(0)

    for i, q in enumerate(questions):
        a_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": q}]
        )
        answer = a_response.choices[0].message.content.strip()

        qa_data.append({"Topic": topic, "Question": q, "Answer": answer})
        progress.progress((i + 1) / len(questions))
        st.markdown(f"**Q{i+1}. {q}**\n\n{answer}\n")

    df = pd.DataFrame(qa_data)
    csv = df.to_csv(index=False).encode("utf-8")

    st.success("✅ Done! Your Q&A dataset is ready.")
    st.download_button(
        label="📥 Download Q&A Dataset as CSV",
        data=csv,
        file_name=f"{topic}_qa_dataset.csv",
        mime="text/csv"
    )
