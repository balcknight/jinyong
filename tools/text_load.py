
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def simple_text_load(file_path='data/倚天屠龙记.txt'):
    loader = TextLoader(file_path, encoding='UTF-8')
    documents = loader.load()  # document对象列表
    return documents

def text_split(file_path,chunk_size=4000,chunk_overlap=2500):
    '''
    :param file_path: 需要切割的文档
    :return: 多个被切好的文档
    '''
    document=simple_text_load(file_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # 每段的字符数
        chunk_overlap=chunk_overlap,  # 重叠的部分
    )
    split_texts = text_splitter.split_documents(document)
    return split_texts



