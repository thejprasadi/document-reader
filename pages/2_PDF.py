__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
#from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
import os
import openai
from langchain_community.document_loaders import PyPDFLoader
import tempfile


os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)

st.title("PDF Summarizer & QA")
pdf_file = st.file_uploader("Choose a PDF file", type="pdf")

if pdf_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf_file.read())
        pdf_path = tmp_file.name
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()

page_selection = st.radio("Page selection", ["Single page", "Page range", "Overall Summary", "Question"])

if page_selection == "Single page":
    page_number = st.number_input("Enter page number", min_value=1, max_value=len(pages), value=1, step=1)
    view = pages[page_number - 1]
    texts = text_splitter.split_text(view.page_content)
    docs = [Document(page_content=t) for t in texts]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summaries = chain.run(docs)

    st.subheader("Summary")
    st.write(summaries)