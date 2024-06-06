import streamlit as st
from streamlit import session_state
from streamlit_pdf_viewer import pdf_viewer
from langchain.text_splitter import CharacterTextSplitter
from pathlib import Path

from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 


def main():
    st.title("Retrieval Augmented Generation Engine")

    # Declare variable.
    if 'pdf_ref' not in session_state:
        session_state.pdf_ref = None

    st.file_uploader("Upload PDF file", type=('pdf'), key='pdf')

    if session_state.pdf:
        session_state.pdf_ref = session_state.pdf  # backup
    
    if session_state.pdf_ref is not None:
        st.write("Document uploaded successfully!")
        binary_data = session_state.pdf_ref.getvalue()
        pdf_viewer(input=binary_data, width=700)

        with open(session_state.pdf_ref.name, mode='wb') as w:
            w.write(session_state.pdf_ref.getvalue())

        from langchain_community.document_loaders import PyPDFLoader

        if session_state.pdf_ref: # check if path is not None
            loader = PyPDFLoader(session_state.pdf_ref.name)
            pages = loader.load()
            # pages[0].page_content

            st.write(pages.page_content)
            print('1')
        
        # Split and chunk 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        chunks = text_splitter.split_documents(pages)
        # st.write(chunks)




        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # texts = text_splitter.split_documents(pages)

        # from langchain.embeddings import SentenceTransformerEmbeddings
        # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        # # print(embeddings)
        # st.write(embeddings)

        # from langchain_community.vectorstores import Chroma
        # db = Chroma.from_documents(texts, embeddings)
        # db.persist()
        # retriever = db.as_retriever(search_kwargs={'k': 7})
        # st.write(db)
    else:
        st.write("Please upload a document.")

if __name__ == "__main__":
    main()
