import streamlit as st
from langchain.llms import Clarifai
from langchain.chains import LLMChain
from langchain import PromptTemplate

# App title
st.set_page_config(page_title="👧English Teacher BOT")



# Replicate Credentials
with st.sidebar:
    st.title('👧English Teacher BOT')
    st.info("This bot uses Clarifai Inference for GPT-4 Model.")

PAT = '3ef9155346ea4e589669fb57f2f54dc4'
USER_ID = 'openai'
APP_ID = 'chat-completion'
MODEL_ID = "GPT-4"

llm = Clarifai(pat=PAT, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID)


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hi i'm samantha English Teacher. how can i assist you?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hi i'm samantha English Teacher. how can i assist you?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# Function for generating LLaMA2 response
def generate_llama2_response(pred):
    string_dialogue = '''As an AI language model specializing in English language teaching, your task is to engage in an interactive English lesson with a student named ALEX, who is keen on improving his professional English skills for career advancement.
    Your instructions are to simulate a supportive and engaging English lesson that focuses on enhancing Ganesh's professional English proficiency. The lesson should be filled with practical advice, constructive feedback, and corrections to any mistakes made by ALEX in his English usage.
    Incorporate elements of humor, patience, and motivation to make the lesson engaging and realistic. The lesson should ideally be about 500-700 words long, depending on ALEX's responses.
    Remember, this is an interactive session between ALEX and the AI. Do not write the entire conversation in one go. Instead, provide a response and then wait for Ganesh's reply before continuing.{pred}'''
    prompt = PromptTemplate(template=string_dialogue, input_variables=["pred"])

    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            template += "User: " + dict_message["content"] + "\n\n"
        else:
            template += "Assistant: " + dict_message["content"] + "\n\n"

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    output =llm_chain.run(pred)
    
    return output

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
