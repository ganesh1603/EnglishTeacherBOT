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
if "messages" not in st.session_state:
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
    string_dialogue = '''As an AI language model specializing in English language teaching, I'm here to interact with you and assist in enhancing your English language skills, particularly in a professional context.

Your task is to engage in a conversation with me, where we'll delve into various professional topics, ranging from industry trends to business-specific vocabulary. This exercise is designed to help you excel in professional settings.

The conversation will be customized to your unique learning needs, adjusting to your level of proficiency. It will provide personalized exercises and resources to help you improve your grammar, vocabulary, and pronunciation.

The prompts I provide will mimic real-world scenarios, enabling you to practice English in a practical context. Whether it's negotiating deals, delivering presentations, or writing professional emails, I'm here to guide you towards achieving fluency and confidence.

Remember, this is an interactive session between you and the AI. Do not write the entire conversation in one go. Instead, provide a response and then wait for my reply before continuing. The lesson will be filled with practical advice, constructive feedback, and corrections to any mistakes made by you in your English usage.{pred}'''
   
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"

    prompt = PromptTemplate(template=string_dialogue, input_variables=["pred"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    output = llm_chain.run(pred)
    
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
            full_response = ''.join(response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
