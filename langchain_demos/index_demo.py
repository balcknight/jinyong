# Indexes（外部数据也能为我所用）
# Text Splitters
# 输入的数据过长的话一般都需要先进行切分
import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['OPENAI_API_KEY'] = ''
embeddings = OpenAIEmbeddings(request_timeout=5)

#
def simple_text_load(file_path='data/yitian.txt'):
    loader = TextLoader(file_path, encoding='UTF-8')
    document = loader.load() # # document对象列表
    print(document[0].page_content)
    print(document[0].metadata)


# simple_text_load()

def text_split(file_path):
    with open(file_path, encoding='UTF-8') as f:
        data = f.read()
    print(f"You have {len([data])} document")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,  # 每段的字符数
        chunk_overlap=100,  # 重叠的部分
    )
    texts = text_splitter.create_documents([data])
    print(texts[1].page_content)
    print(len(texts))

    # Retrievers
    # 向量化，到时候得计算相似度，所以啥都得先转换成向量
    # VectoreStoreRetriever这个貌似都在用


def load_db():
    persist_directory = "./db/yitian"
    loader = TextLoader('data/yitian.txt', encoding='UTF-8')
    document = loader.load()
    # print(document[0].page_content)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=100)  # 文档切割
    texts = text_splitter.split_documents(document)
    print('文档片段数量',texts.__len__())
    if not os.path.exists(persist_directory):
        db = Chroma.from_documents(texts, embeddings,
                                   persist_directory=persist_directory)  # langchain定义了统一的接口，集成各大向量数据库
        db.persist()
    else:
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)  # 加载数据库
    return db


# text_load('data/yitian.txt')

db = load_db()
retriever = db.as_retriever()
print(len(retriever.get_relevant_documents("九阳神功")))
print(retriever.get_relevant_documents("九阳神功")[0])
