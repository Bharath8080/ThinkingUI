import os
import base64
import streamlit as st
from dotenv import load_dotenv
from ollama import Client

# Load environment variables
load_dotenv()

# ✅ Initialize Ollama client
client = Client(
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
)

# Streamlit page setup
st.set_page_config(page_title="Ollama Thinking Chat", layout="centered")

def process_thinking_stream(stream):
    """Process streamed response showing thinking and final output."""
    thinking_content = ""
    response_content = ""

    # Thinking phase with progress status
    with st.status("💭 Thinking...", expanded=True) as status:
        for part in stream:
            message = part.get("message", {})
            if "thinking" in message:
                thinking_content += message["thinking"]
            elif "content" in message:
                response_content += message["content"]
        status.update(label="✅ Thinking complete!", state="complete", expanded=False)
    
    return thinking_content, response_content


def display_assistant_message(content, thinking_content=None):
    """Display assistant's thinking and final message."""
    if thinking_content and thinking_content.strip():
        with st.expander("🧠 Thinking process", expanded=False):
            st.markdown(thinking_content)
    if content:
        st.markdown(content)


def display_message(message):
    """Display each chat message."""
    role = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(role):
        if role == "assistant":
            display_assistant_message(
                message.get("content"),
                message.get("thinking")
            )
        else:
            st.markdown(message["content"])


def display_chat_history():
    """Display chat history."""
    for message in st.session_state["messages"]:
        if message["role"] != "system":
            display_message(message)


@st.cache_resource
def get_chat_model():
    """Return a callable chat model for streaming."""
    def model(messages):
        # ✅ Use Ollama API for streaming
        return client.chat('minimax-m2:cloud', messages=messages, stream=True)
    return model


def handle_user_input():
    """Handle user input and generate streaming assistant response."""
    if user_input := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # Display user input
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display assistant response with streaming
        with st.chat_message("assistant"):
            chat_model = get_chat_model()
            stream = chat_model(st.session_state["messages"])

            # Process streaming output
            thinking_content, response_content = process_thinking_stream(stream)

            # Display formatted output
            display_assistant_message(response_content, thinking_content)

            # Save in history
            st.session_state["messages"].append({
                "role": "assistant",
                "content": response_content,
                "thinking": thinking_content
            })


def main():
    """Main chat layout."""
    minimax_logo = base64.b64encode(open("assets/minimax.png", "rb").read()).decode()
    ollama_logo = base64.b64encode(open("assets/ollama.png", "rb").read()).decode()

    st.markdown(f"""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2 style='margin-bottom: 1rem;'>
            <img src="data:image/png;base64,{ollama_logo}" width="40" style="vertical-align: middle; margin-right: 10px;">
            Ollama Minimax-M2:Cloud Chat
            <img src="data:image/png;base64,{minimax_logo}" width="40" style="vertical-align: middle; margin-left: 10px;">
        </h2>
        <h4 style='color: #666; margin-top: 0;'>With thinking UI! 💡</h4>
    </div>
    """, unsafe_allow_html=True)

    display_chat_history()
    handle_user_input()


if __name__ == "__main__":
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": "You are a helpful AI assistant that explains your reasoning before answering."}
        ]
    main()
