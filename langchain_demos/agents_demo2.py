from langchain.tools.retriever import create_retriever_tool
import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['OPENAI_API_KEY'] = ''
embeddings = OpenAIEmbeddings()

def vectorstore_use():
    persist_directory = "./db/yitian"
    loader = TextLoader('data/yitian.txt', encoding='UTF-8')
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)  # 文档切割
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(request_timeout=5)
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)  # langchain定义了统一的接口，集成各大向量数据库

    retriever = db.as_retriever()  # 创建检索器，数据库是后端，可以使用多种数据库
    return retriever

retriever=vectorstore_use()

retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)