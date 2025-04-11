from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from create_database import add_to_chroma
from secret import DATA_PATH

def load_documents():
    document_loader=  PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents:list[Document],chunk:int):
    pass
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size=chunk,
        chunk_overlap = chunk/10,
        is_separator_regex=False,
        length_function = len,
    )
    return text_splitter.split_documents(documents)

documents = load_documents()
chunks = split_documents(documents,800)
add_to_chroma(chunks)