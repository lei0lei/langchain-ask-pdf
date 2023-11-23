from dotenv import load_dotenv
import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
# from langchain.callbacks import get_openai_callback


import os

#
from streamlit_extras.row import row 
from streamlit_chat import message


# def _x_main():
#     load_dotenv()
#     st.set_page_config(page_title="Ask your PDF")
#     st.header("Ask your PDF 💬")
    
#     # upload file
#     pdf = st.file_uploader("Upload your PDF", type="pdf")
    
#     # extract the text
#     if pdf is not None:
#       pdf_reader = PdfReader(pdf)
#       text = ""
#       for page in pdf_reader.pages:
#         text += page.extract_text()
        
#       # split into chunks
#       text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#       )
#       chunks = text_splitter.split_text(text)
      
#       # create embeddings
#       embeddings = OpenAIEmbeddings()
#       knowledge_base = FAISS.from_texts(chunks, embeddings)
      
#       # show user input
#       user_question = st.text_input("请询问:")
#       if user_question:
#         docs = knowledge_base.similarity_search(user_question)
        
#         llm = OpenAI()
#         chain = load_qa_chain(llm, chain_type="stuff")
#         with get_openai_callback() as cb:
#           response = chain.run(input_documents=docs, question=user_question)
#           print(cb)
           
#         st.write(response)
    

def _main():
  st.header("Helper for your pdf paper")
  st.markdown('<style>' + open('./frontend/style.css').read() + '</style>', unsafe_allow_html=True)


  with st.sidebar:
      tabs = on_hover_tabs(tabName=['Settings','MainPage'], 
                          iconName=['settings','home'], default_choice=0)

    # st.write(f'select model')
    # st.write(f'set model params')
    # st.write(f'shown windows')
  
  if tabs =='Settings':
      st.title("Setting page")
      st.write('Name of option is {}'.format(tabs))
      add_selectbox = st.sidebar.selectbox(
          'select model',
          ('chatgpt3.5', 'chatgpt4')
      )

  elif tabs == 'MainPage':
      pass
# =====================================================================================================
BACKGROUND_COLOR = 'white'
COLOR = 'black'

def __set_page_container_style(
        max_width: int = 1100, max_width_100_percent: bool = False,
        padding_top: int = 1, padding_right: int = 10, padding_left: int = 1, padding_bottom: int = 10,
        color: str = COLOR, background_color: str = BACKGROUND_COLOR,
    ):
        if max_width_100_percent:
            max_width_str = f'max-width: 100%;'
        else:
            max_width_str = f'max-width: {max_width}px;'
        st.markdown(
            f'''
            <style>
                .reportview-container .sidebar-content {{
                    padding-top: {padding_top}rem;
                }}
                .reportview-container .main .block-container {{
                    {max_width_str}
                    padding-top: {padding_top}rem;
                    padding-right: {padding_right}rem;
                    padding-left: {padding_left}rem;
                    padding-bottom: {padding_bottom}rem;
                }}
                .reportview-container .main {{
                    color: {color};
                    background-color: {background_color};
                }}
            </style>
            ''',
            unsafe_allow_html=True,
        )

def __gen_chatgpt_response():
   pass

def __clear_chat_history():
  st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
  
def __main():
  st.set_page_config(page_title="Little Helper",layout='wide')
  st.header("Little Helper for your pdf paper")
  st.divider()
  # 设置栏
  with st.sidebar:
    add_title = st.write(
        "Settings",
    )
    if 'OPENAI_API_KEY' in st.secrets:
      st.success('API key already provided!', icon='✅')
      openai_api_key = st.secrets['OPENAI_API_KEY']
    else:
      openai_api_key = st.text_input('Enter Replicate API token:', type='password')
      if not len(openai_api_key)==40:
        st.warning('Please enter your credentials!', icon='⚠️')
      else:
        st.success('Proceed to entering your prompt message!', icon='👉')
        os.environ['OPENAI_API_KEY'] = openai_api_key
    with st.container():
      row_settings_model = row(1, vertical_align="top")
      row_settings_model.selectbox("Select model", ["gpt3.5", "gpt4"])
      row_settings_t = row(1, vertical_align="top")
      temperature = row_settings_t.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
      row_settings_save = row(1, vertical_align="top")
      row_settings_save.button("Save", use_container_width=True)
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
  # 主页面
  st.markdown(
        f"""
    <style>
        .appview-container .main .block-container{{
                padding-top: {3}rem;    }}
        </style>
    """,
        unsafe_allow_html=True,
    )
  
  if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
  # Display or clear chat messages
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

  
  openai_api_key = '1'
  if prompt := st.chat_input(disabled=not openai_api_key):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
      st.write(prompt)

  main_container = st.container()
  pdf_col,  summary_col = main_container.columns([2,1])
  pdf = pdf_col.file_uploader("Upload your PDF", type="pdf")

  with summary_col:
    title_summary_row = row(1, vertical_align="top")
    title_summary_row.write(f'summary')



def main():
    st.set_page_config(page_title="Little Helper",layout='centered')

    st.header("Little Helper ")
    st.divider()
  
    st.markdown(
        f"""
    <style>
        .appview-container .main .block-container{{
                padding-top: {3}rem;    }}
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.write(f'help doc here')


if __name__ == '__main__':
    main()
