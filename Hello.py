import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import openai
import tempfile

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Research Application" ,layout="wide", initial_sidebar_state="auto")

st.title('Text Reader')

template = """Answer the question based only on the following context:

{context}
If you don't know the answer, just say out of scope, don't try to make up an answer.


Question: {question}
"""


# uploaded_file = st.file_uploader("Add text file !")
# if uploaded_file:
#     for line in uploaded_file:
#         st.write(line)

uploaded_file = st.file_uploader("Choose a Text file", type="txt")

def generate_response():
    if uploaded_file:
        documents = [uploaded_file.read().decode()]
    # st.write(documents)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
       
        return(texts)


texts= generate_response()  

def set_chain(qa):
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()

   
)
    return chain.invoke(qa)


def format_docs(docs):

    format_D="\n\n".join([d.page_content for d in docs])
    
    return format_D






st.markdown("""
<style>
    /* Target the buttons inside the Streamlit app to expand to full width */
    .stButton>button {
        width: 100%;
    }
            
    
    
</style>
""", unsafe_allow_html=True)

import streamlit as st




col1, col2, col3 = st.columns([3, 1, 3])


with col1:
    user_input = st.text_area("Enter Your Query Here", height=300)

with col2:
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    submit_btn = st.button("Submit", key="summary_btn")
    


def submit(text):
    return set_chain(text)



with col3:

    if submit_btn:
      with st.spinner('Submiting Query...'):
        query_result = submit(user_input)
        st.text_area("Query Output", value=query_result, height=300, key='result')

    else:
      st.text_area("Result", height=300, key='result')






    
    


   
       




