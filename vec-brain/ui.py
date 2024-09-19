import os
import re

import streamlit as st
from assistant import Assistant

st.title("Vec Brain")

style_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(style_path, "r") as f:
    style = f.read()

st.markdown(
    f"""
<style>
{style}
</style>
""",
    unsafe_allow_html=True,
)


# Initialize assistant (including chat history)
if "assistant" not in st.session_state:
    st.session_state.assistant = Assistant()

# Display chat messages from history on app rerun
for message in st.session_state.assistant.chat_history.messages:
    role = "assistant" if message.type == "ai" else "user"
    with st.chat_message(role):
        st.markdown(message.content)

# Accept user input
if prompt := st.chat_input("What do you want to learn about today?"):

    st.session_state.assistant.chat_history.add_user_message(prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    # load docs from URL
    if "http" in prompt:
        documents = []
        for url in re.findall(r"(https?://\S+)", prompt):
            documents += st.session_state.assistant.load_from_url(url)
        st.session_state.assistant.vectorstore.add_documents(documents)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = st.session_state.assistant.chain.stream(
            {
                "messages": st.session_state.assistant.chat_history.messages,
            }
        )

        response = st.write_stream(stream)
        st.session_state.assistant.chat_history.add_ai_message(response)
