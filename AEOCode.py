import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import os
import re

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Streamlit App UI
st.set_page_config(page_title="Auto Q&A Generator", page_icon="🤖", layout="wide")
st.title("🤖 Automatic Q&A Generator with Sources")

# Check for API key
if not api_key:
    st.error("⚠️ OPENAI_API_KEY not found. Please create a `.env` file in the same folder with your key.")
    st.stop()

client = OpenAI(api_key=api_key)

# Sidebar inputs
st.sidebar.header("⚙️ Settings")
topic = st.sidebar.text_input("Enter a topic for Q&A generation:")
num_questions = st.sidebar.slider("Number of questions to generate", 3, 15, 5)

generate_btn = st.sidebar.button("🚀 Generate Q&A")

if generate_btn and topic.strip():
    st.subheader(f"🧠 Generating {num_questions} Q&A for: **{topic}** ...")
    
    # Step 1: Generate Questions
    q_prompt = f"Generate {num_questions} diverse and interesting questions about {topic}. Only return the list of questions."
    try:
        q_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": q_prompt}]
        )
        raw_questions = q_response.choices[0].message.content.strip().split("\n")
        questions = [q.strip(" -0123456789.").strip() for q in raw_questions if q.strip()]
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        st.stop()

    # Step 2: Generate Answers (with Sources)
    qa_data = []
    for i, q in enumerate(questions):
        st.markdown(f"**Q{i+1}. {q}**")
        
        a_prompt = f"""
        {q}

        Please answer clearly and concisely.
        Include 2–3 credible sources or references (with URLs if available)
        under a heading 'Sources:' at the end.
        """

        try:
            a_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": a_prompt}]
            )
            answer = a_response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"Error generating answer: {e}"

        # Try to extract sources from the answer
        sources = ""
        if "Sources:" in answer or "References:" in answer:
            parts = re.split(r"Sources:|References:", answer)
            if len(parts) > 1:
                sources = parts[-1].strip()

        # Display answer and sources
        st.write(answer)
        if sources:
            st.markdown(f"📚 **Sources:**\n\n{sources}")

        qa_data.append({
            "Topic": topic,
            "Question": q,
            "Answer": answer,
            "Sources": sources
        })

    # Step 3: Downloadable CSV
    st.success("✅ Q&A generation complete!")
    df = pd.DataFrame(qa_data, columns=["Topic", "Question", "Answer", "Sources"])
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Q&A Dataset (CSV)",
        data=csv,
        file_name=f"{topic.lower().replace(' ', '_')}_qa_dataset.csv",
        mime="text/csv"
    )

elif generate_btn and not topic.strip():
    st.warning("Please enter a topic before generating Q&A.")
