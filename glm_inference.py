import logging
from concurrent.futures import ThreadPoolExecutor
import os

import glob
from random import random

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from llm_study.jinyong.tools.query_agents import get_answer_by_google, get_answer_by_rag
from tools.prompt_temp import get_react_answer_messages, get_rag_messages
from openai import OpenAI

base_url = "http://127.0.0.1:8000/v1/"
llm = ChatOpenAI(openai_api_key="EMPTY", base_url=base_url,temperature=.0, max_tokens=8196,request_timeout=20)
logging.basicConfig(filename='error-2.log', encoding='utf-8', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# client = OpenAI(api_key="EMPTY", base_url=base_url)

# print(chat.invoke("你是谁？").content)

def simple_chat(input_content):
    messages = [
        {
            "role": "system",
            "content": "你是一个专业的新闻学家，请生成下面文章的新闻标题，只需要输出标题，不要其他内容",
        },
        {
            "role": "user",
            "content": input_content
        }
    ]
    response = client.chat.completions.create(
        model="chatglm3-6b",
        messages=messages,
        max_tokens=512,
        temperature=0.4,
        presence_penalty=1.1,
        top_p=0.8)

    content = response.choices[0].message.content
    return content

def get_react_questions(input_content):
    messages = get_react_answer_messages(input_content)
    res = llm.invoke(messages).content
    return res


def get_answer_by_react(input_content):
    sub_questions = get_react_questions(input_content)
    # sub_questions = '1. 慕容复是金庸哪部小说中的角色？\n'
    dict_list = []
    for sub_question in sub_questions.split('\n'):
        print(sub_question)
        # 随机选择一种方法获取答案
        # methods = [get_answer_by_google, get_answer_by_rag, get_answer_by_wiki]
        methods = {
            get_answer_by_google: "Google",  # 欠费了,
            # get_answer_by_wiki:"Wiki",
            get_answer_by_rag: "RAG"
        }
        answers = []
        try:
            chosen_method = random.choice(list(methods.keys()))

            answer = chosen_method(sub_question)
            answers.append(answer)
        except Exception as e:
            print('异常信息:', e)
        # 得到子问题的答案
        rag_infos=get_rag_messages(input_content,answers)
        res=llm.invoke(rag_infos)
        return res

print(get_answer_by_react("《神雕侠侣》的结局是什么？"))
