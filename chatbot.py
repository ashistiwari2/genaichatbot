import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
OPENAI_API_KEY='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'
OPENAI_API_KEY=st.secret['OPENAI_API_KEY']
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
        llm= ChatOpenAI(openai_api_key=OPENAI_API_KEY,temperature=0,max_tokens=1000,model_name='gpt-3.5-turbo')
        chain=load_qa_chain(llm,chain_type="stuff")
        response=chain.run(input_documents=match,question=user_question)
        st.write(response)
