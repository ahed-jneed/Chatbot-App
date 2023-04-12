from streamlit.server import Server
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def get_session_info(session_id):
    if session_id is None:
        return None
    session_info = Server.get_current()._session_info_by_id.get(session_id)
    return session_info

def get_session():
    report_ctx = st.report_thread.get_report_ctx()
    session_id = None
    if report_ctx is not None:
        session_id = report_ctx.session_id

    session_info = get_session_info(session_id)
    if session_info is None:
        session = SessionState()
        session.request_rerun = True
    else:
        if "session" not in session_info:
            session = SessionState()
            session_info["session"] = session
        else:
            session = session_info["session"]
    return session

st.session_state = get_session()

st.set_page_config(page_title='🤖Your Talent coach🤖', layout='wide')

if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

def get_text():
    """
    Get the user input text.
    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Your AI assistant here! Ask me anything ...",
                            label_visibility='hidden')
    return input_text

def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.store = {}
    st.session_state.entity_memory.buffer.clear()

# Additional feature : Start a new chat
with st.sidebar.expander(" 🛠️ Settings ", expanded=False):
    # Option to preview memory store
    if st.checkbox("Preview memory store", key="checkbox1"):
        st.write(st.session_state.entity_memory.store)
    # Option to preview memory buffer
    if st.checkbox("Preview memory buffer"):
        st.write(st.session_state.entity_memory.buffer)
    MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo','text-davinci-003','text-davinci-002','code-davinci-002'])
    K = st.number_input(' (#)Summary of prompts to consider',min_value=3,max_value=1000)



# st.markdown(" > Powered by -  🦜 LangChain + OpenAI + Streamlit")

# Ask the user to enter their OpenAI API key
API_O = st.sidebar.text_input(":blue[Enter Your OPENAI API-KEY :]",
                placeholder="Paste your OpenAI API key here (sk-...)",
                type="password") # Session state storage would be ideal

if API_O:
    # Create an OpenAI instance
    llm = OpenAI(temperature=0,
                 openai_api_key=API_O,
                 model_name=MODEL,
                 verbose=False)

    # Create a ConversationEntityMemory object if not already created
    if 'entity_memory' not in st.session_state:
        st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K)

    # Create the ConversationChain object with the specified configuration
    Conversation = ConversationChain(
        llm=llm,
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory=st.session_state.entity_memory
    )
else:
    st.markdown(''' 
        ```
        - 1. Enter API Key + Hit enter 🔐 

        - 2. Ask anything via the text input widget

        Your API-key is not stored in any form by this app. However, for transparency ensure to delete your API once used.
        ```

        ''')
    st.sidebar.warning('API key required to try this app.The API key is not stored in any form.')
    # st.sidebar.info("Your API-key is not stored in any form by this app. However, for transparency ensure to delete your API once used.")

# Implementing a Button to Clear the memory
st.sidebar.button("New Chat", on_click = new_chat, type='primary')

#Get the user INPUT and RUN the chain. Also, store them
user_input = get_text()
if user_input:
    output = Conversation.run(input=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

# Display the conversation history using an expander
# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i], icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])

    # Can throw error - requires fix
    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download', download_str)

# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label= f"Conversation-Session:{i}"):
            st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session
