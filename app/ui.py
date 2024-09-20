import os

import streamlit as st
from assistant import Assistant

st.title("Vec Brain")

cwd = os.path.dirname(__file__)
style_path = os.path.join(cwd, "style.css")
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

asset_path = os.path.join(cwd, "assets/")
user_icon = os.path.join(asset_path, "user_icon.jpg")
ai_icon = os.path.join(asset_path, "ai_icon.jpg")

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

    st.session_state.assistant.add_user_message_with_documents(prompt)

    with st.chat_message("user", avatar=user_icon):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar=ai_icon):
        stream = st.session_state.assistant.chain.pick("answer").stream(
            {
                "messages": st.session_state.assistant.chat_history.messages,
            }
        )

        response = st.write_stream(stream)
        st.session_state.assistant.chat_history.add_ai_message(response)
