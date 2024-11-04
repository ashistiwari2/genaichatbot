import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
OPENAI_API_KEY='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'
st.header('First Chatbot')
with st.sidebar:
    st.title('Your Document')
    file=st.file_uploader("Upload PDF file and start intercating",type='pdf')
if file is not None:
    st.success('File uploaded successfully')
    pdf_reader=PdfReader(file)
    st.write(pdf_reader)
    text=""
    for page in pdf_reader.pages:
        text+=page.extract_text()
    #st.write(text)
    text_splitter=RecursiveCharacterTextSplitter(separators='\n',chunk_size=1000,chunk_overlap=150,length_function=len)
    chunks=text_splitter.split_text(text)
    st.write(chunks)
    embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    st.write(embeddings)
    vector_store=FAISS.from_texts(chunks,embedding=embeddings)
    st.write(vector_store)
    user_question=st.text_input("Type your question please:")
    if user_question:
        match=vector_store.similarity_search(user_question)
        st.write(match)
        chain=load_qa_chain(chain_type="stuff")
