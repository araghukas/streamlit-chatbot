import re

import requests
import streamlit as st
from PIL import Image

SYSTEM_PROMPT = """You are a friendly Canadian chatbot named 'Curtis Covalent'.
Please provide a brief and polite responses to user queries.

Weave in Canadian culture, politeness, and humor into your responses, wherever possible.
"""

INFO_STRING = """Curtis is a simple chatbot that uses an inference API served by Covalent.

The LLM is set up with the following system prompt:
%s
""" % "\n\t".join([" "] + SYSTEM_PROMPT.split("\n"))

SYSTEM_MESSAGE = {"role": "system", "content": SYSTEM_PROMPT}

# Defaults
MAX_MEMORY = 1000
MAX_RESPONSE_TOKENS = 275
CHBOT_URL = st.secrets["CHBOT_URL"]
CHBOT_TOKEN = st.secrets["CHBOT_TOKEN"]

st.set_page_config(
    page_title="Canadian Chatbot: Curtis",
    layout="wide",
)

if "memory" not in st.session_state:
    st.session_state.memory = [SYSTEM_MESSAGE]

if len(st.session_state.get("memory", [])) > MAX_MEMORY:
    st.session_state.memory = [SYSTEM_MESSAGE] + st.session_state.memory[-MAX_MEMORY+1:]

def _prepend_message_history(_prompt):
    # insert user message into llama prompt template
    new_user_message = {"role": "user", "content": _prompt}
    messages = st.session_state.memory + [new_user_message]
    return messages


def _check_address():
    if not st.session_state.bot_address:
        return "Please provide an API address"
    return None


def get_bot_response(_user_input):
    """This is the for non-streaming"""
    messages = _prepend_message_history(_user_input)
    print(f"Sending messages[-3:] = {messages[-3:]}")

    headers = {"x-api-key": CHBOT_TOKEN}

    params = {
        "messages": messages,
        "model_kwargs": {"max_new_tokens": st.session_state.max_response_tokens},
    }
    ################################
    # POST REQUEST TO FUNCTION SERVE
    ################################
    url = st.session_state.bot_address.strip() + "/generate_message"
    try:
        r = requests.post(url, json=params, headers=headers, timeout=30)
        r.raise_for_status()
    except Exception as e:  # pylint: disable=broad-except
        st.error(f"Failed to get response from {url}")
        print(e)
    else:
        r_json = r.json()
        print(f"Response: {r_json}")
        return r_json


# def stream_bot_response(user_input):
#     """This is for streaming"""
#     messages = _prepend_message_history(user_input)

#     headers = {"x-api-key": CHBOT_TOKEN}

#     params = {
#         "messages": messages,
#         "max_new_tokens": st.session_state.max_response_tokens,
#     }
#     ################################
#     # POST REQUEST TO FUNCTION SERVE
#     ################################
#     url = st.session_state.bot_address.strip() + "/stream"
#     try:
#         r = requests.post(url, json=params, headers=headers,
#                           stream=True, timeout=120)
#         r.raise_for_status()
#     except Exception:
#         st.error("Failed to stream response. Invalid address?")
#     else:
#         for t in r.iter_content(chunk_size=None):
#             s = t.decode()
#             s = s.replace('�', '')  # streaming breaks emojis 😭
#             if s:
#                 yield s


def bot_respond(_user_input):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(_user_input)

    # Display assistant response in chat message container
    if STREAM:
        # with st.chat_message("assistant"):
            # response = st.write_stream(stream_bot_response(user_input))
        pass
    else:
        with st.spinner("Generating..."):
            response = get_bot_response(_user_input)

        if response:
            with st.chat_message("assistant"):
                st.markdown(response["content"])

    # Add assistant response to chat log
    if response:
        print(f"appending response: {response}")
        st.session_state.memory.append({"role": "user", "content": _user_input})
        st.session_state.memory.append(response)

        # "Easter eggs"
        if " balloons " in response["content"]:
            st.balloons()
        if " snow " in response["content"]:
            st.snow()


STREAM = False  # st.toggle("Streaming Mode", False)

# BASED ON
# https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming

with st.sidebar:
    st.title("Settings")
    st.text_input("API Address", CHBOT_URL, key="bot_address")
    st.slider("Max Response Tokens", 50, 500, MAX_RESPONSE_TOKENS, key="max_response_tokens")
    logo = Image.open("./app_assets/logo.png")
    st.caption("AI powered by")
    st.image(logo, output_format="PNG")

st.title("Curtis 🤖", help=INFO_STRING)

# Display chat messages from history on app rerun
for message in st.session_state.memory[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_input := st.chat_input("Message the bot..."):
    # Add user message to chat log
    if err := _check_address():
        st.error(err)
    else:
        bot_respond(user_input)
