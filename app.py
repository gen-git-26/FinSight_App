import streamlit as st
from ango_agent import AngoAgent  # Make sure you have ango_agent installed and available
from agent.agent import build_agent  # Import your Agent class if needed

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize AngoAgent (customize tools as needed)
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()

st.title("Ango Agent Chat")

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Ango Agent:** {msg['content']}")

# User input
user_input = st.text_input("Type your message:", key="input")

if st.button("Send") and user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get agent response
    response = st.session_state.agent.chat(user_input)
    st.session_state.messages.append({"role": "agent", "content": response})

    # Rerun to display new messages
    st.experimental_rerun()
