import os

from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from zhipuai import ZhipuAI

from langchain_glm.ZhipuChat import ChatZhipuAI
from messages.novel_enum import Jinyong
from tools.db_load import load_retriever, get_rag_text_by_retriever
from tools.prompt_temp import get_rag_messages

API_KEY = ''
os.environ["SERPER_API_KEY"] = ""

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

llm = ChatZhipuAI(api_key=API_KEY, model='glm-4', max_tokens=8192, top_p=0.5, timeout=240, stream=False)

search = GoogleSerperAPIWrapper(lang='zh',k=5)
api_wrapper = WikipediaAPIWrapper(top_k_results=4, lang='zh', doc_content_chars_max=10000)

def wikiApi_query(query_key):
    # 工具包含 name describe 输入内容的json格式， 调用函数 结果是否直接返回给用户
    tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    res = tool.run(query_key)
    return res


def google_query(query_key):
    res=search.run(query_key)
    return res

def get_answer_by_rag(user_question, novel_name=Jinyong.YiTian.value):
    retriever = load_retriever(db_name=novel_name)
    rag_text = get_rag_text_by_retriever(retriever, user_question)
    rag_messages=get_rag_messages(novel_name,user_question,rag_text)
    return llm.invoke(rag_messages).content


def get_answer_by_google(user_question, novel_name=Jinyong.YiTian.value):
    rag_text = google_query(user_question)
    rag_messages = get_rag_messages(novel_name, user_question, rag_text)
    return llm.invoke(rag_messages).content
