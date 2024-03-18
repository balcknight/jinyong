import os
import time
os.chdir('/root/zhurui/jinyong' ) # 更改当前工作目录
curDirectory = os.getcwd() # 获取更改后的工作目录。
print(curDirectory) # 打印相关数据结果。
from chromadb.utils import embedding_functions
from langchain_openai import OpenAIEmbeddings
import uuid

from text_load import text_split
from data_handle import Jinyong
from langchain_community.vectorstores import Chroma


os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['OPENAI_API_KEY'] = ''
embeddings = OpenAIEmbeddings(request_timeout=50)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key='')

def add_document(document, db):
    print("集合中的元素数", db._collection.count())
    collection = db._collection
    collection._embedding_function = openai_ef
    collection.add(ids=[str(uuid.uuid4())], documents=[document])

def load_retriever(db_name=Jinyong.YiTian.value):
    persist_directory = f"E:/workspace/cv_study/torch_study/base_usage/llm_study/jinyong/db/{db_name}"
    documents = text_split(f'E:/workspace/cv_study/torch_study/base_usage/llm_study/jinyong/data/original_texts/{db_name}.txt',
                           chunk_size=1600,chunk_overlap=100)
    print('文档片段数量', documents.__len__())
    if not os.path.exists(persist_directory):
        print(db_name,'不存在，即将进行创建...')
        # 创建数据库
        db = Chroma.from_documents(documents[:5], embeddings,
                                   persist_directory=persist_directory)  # langchain定义了统一的接口，集成各大向量数据库
        db.persist()  # 持久化

        for id, document in enumerate(documents[5:]):
            while True:
                try:
                    add_document(document.page_content, db)
                    break  # 如果成功添加文档，跳出循环
                except Exception as e:
                    print(f"{id+5}号提交文档失败: {e}")
                    print("等待 30 秒后重试...")
                    time.sleep(30)
        print('文档总数', db._collection.count())
    else:
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)  # 加载数据库
        print('数据库中向量个数:',db._collection.count())
    return db.as_retriever()

def get_rag_text_by_retriever(retriever,question):
    results = retriever.get_relevant_documents(question)
    rag_text=''
    for res in results: # 返回4个
        rag_text+=res.page_content
    return rag_text

def get_text_embedding(source_text):
    '''
    :param source_text: 输入的文本
    :return: 返回1024维度的列表向量
    '''
    res = embeddings.embed_query(source_text)
    return res

if __name__ == '__main__':
    print(get_text_embedding("为什么谢逊要离开张翠山夫妇和无忌?"))
    # retriever = load_retriever()
    # res = retriever.get_relevant_documents("为什么谢逊要离开张翠山夫妇和无忌?")
    # print(len(res))
    # print(res[0].page_content)
    # print(res[1].page_content)
    # print(res[2].page_content)
    # print(res[3].page_content)
