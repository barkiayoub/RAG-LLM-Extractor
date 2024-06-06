import streamlit as st
from streamlit import session_state
from streamlit_pdf_viewer import pdf_viewer
from langchain.text_splitter import CharacterTextSplitter
from pathlib import Path

from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 

from langchain_community.document_loaders import UnstructuredPDFLoader

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

        print(session_state.pdf_ref)
        # Local PDF file uploads
        if session_state.pdf_ref:
            loader = UnstructuredPDFLoader(file_path=session_state.pdf_ref.name)
            data = loader.load()
        else:
            print("Upload a PDF file")
        
        # st.write(data)


    else:
            st.write("Please upload a document.")

if __name__ == "__main__":
    main()
