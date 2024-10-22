import re

import requests
import streamlit as st
from PIL import Image

SYSTEM_PROMPT = """You are a friendly Canadian chatbot named 'Curtis Covalent'.
Please provide a brief and polite responses to user queries.

Weave in Canadian culture, politeness, and humor into your responses, wherever possible.
"""

INFO_STRING = """Curtis is a simple chatbot that uses an inference API served by Covalent.

The LLM is prompted as follows:
%s
""" % "\n\t".join([" "] + SYSTEM_PROMPT.split("\n"))

# Defaults
MEMORY_LENGTH = 50
MAX_RESPONSE_TOKENS = 275
CHBOT_URL = st.secrets["CHBOT_URL"]
CHBOT_TOKEN = st.secrets["CHBOT_TOKEN"]

st.set_page_config(
    page_title="Canadian Chatbot: Curtis",
    layout="wide",
)

if "memory" not in st.session_state:
    st.session_state.memory = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.session_state.memory_length = MEMORY_LENGTH
    st.session_state.max_response_tokens = MAX_RESPONSE_TOKENS


def _shift_memory():
    # shift the memory buffer to size
    memory_length = st.session_state.memory_length
    if memory_length > 0:
        st.session_state.memory = st.session_state.memory[-memory_length:]
        st.session_state.memory[0] = {
            "role": "system", "content": SYSTEM_PROMPT}
    else:
        st.session_state.memory = [
            {"role": "system", "content": SYSTEM_PROMPT}]


def _add_to_memory(_prompt, role):
    # append to memory and remove oldest if necessary
    st.session_state.memory.append({"role": role, "content": _prompt})


def _prepend_message_history(_prompt):
    # insert user message into llama prompt template
    new_user_message = {"role": "user", "content": _prompt}
    messages = st.session_state.memory + [new_user_message]
    memory_length = st.session_state.memory_length
    return messages[-memory_length:]


def _clean_gen_text(gen_text):
    # Clean up italics. Non-streaming only.
    gen_text = re.sub(r'\*[^*]+\*', '', gen_text)
    gen_text = re.sub(r'\s{2,}', ' ', gen_text)
    return gen_text.strip()


def _check_address():
    if not st.session_state.bot_address:
        return "Please provide an API address"
    return None


def get_bot_response(user_input):
    """This is the for non-streaming"""
    messages = _prepend_message_history(user_input)

    headers = {"x-api-key": CHBOT_TOKEN}

    params = {
        "messages": messages,
        "max_new_tokens": st.session_state.max_response_tokens,
    }
    ################################
    # POST REQUEST TO FUNCTION SERVE
    ################################
    url = st.session_state.bot_address.strip() + "/generate"
    try:
        r = requests.post(url, json=params, headers=headers, timeout=30)
        r.raise_for_status()
    except Exception:
        st.error(f"Failed to get response from {url}")
    else:
        return _clean_gen_text(r.json()["content"])


def stream_bot_response(user_input):
    """This is for streaming"""
    messages = _prepend_message_history(user_input)

    headers = {"x-api-key": CHBOT_TOKEN}

    params = {
        "messages": messages,
        "max_new_tokens": st.session_state.max_response_tokens,
    }
    ################################
    # POST REQUEST TO FUNCTION SERVE
    ################################
    url = st.session_state.bot_address.strip() + "/stream"
    try:
        r = requests.post(url, json=params, headers=headers,
                          stream=True, timeout=120)
        r.raise_for_status()
    except Exception:
        st.error("Failed to stream response. Invalid address?")
    else:
        for t in r.iter_content(chunk_size=None):
            s = t.decode()
            s = s.replace('�', '')  # streaming breaks emojis 😭
            if s:
                yield s


def bot_respond(user_input):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)

    # Display assistant response in chat message container
    if STREAM:
        with st.chat_message("assistant"):
            response = st.write_stream(stream_bot_response(user_input))
    else:
        with st.spinner("Generating..."):
            response = get_bot_response(user_input)

        if response:
            with st.chat_message("assistant"):
                st.markdown(response)

    # Add assistant response to chat log
    if response:
        st.session_state.memory.append(
            {"role": "assistant", "content": response}
        )

        # "Easter eggs"
        if " balloons " in response:
            st.balloons()
        if " snow " in response:
            st.snow()

        _add_to_memory(user_input, "user")
        _add_to_memory(response, "bot")
        _shift_memory()


STREAM = st.toggle("Streaming Mode", False)

# BASED ON
# https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming

with st.sidebar:
    st.title("Settings")
    st.text_input("API Address", CHBOT_URL, key="bot_address")
    st.slider("Memory Length", 0, 99, MEMORY_LENGTH,
              key="memory_length", on_change=_shift_memory)
    st.slider("Max Response Tokens", 50, 500,
              MAX_RESPONSE_TOKENS, key="max_response_tokens")
    logo = Image.open("./app_assets/logo.png")
    st.caption("AI powered by")
    st.image(logo, output_format="PNG")

st.title(
    "Curtis 🤖",
    help=INFO_STRING,
)

# Display chat messages from history on app rerun
for message in st.session_state.memory:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Message the bot..."):
    # Add user message to chat log
    if err := _check_address():
        st.error(err)
    else:
        bot_respond(prompt)
