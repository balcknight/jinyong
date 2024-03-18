import json
import logging
import os
import random

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langchain_glm.ZhipuChat import ChatZhipuAI
from messages.novel_enum import Jinyong, QuestionType
from tools.data_handle import ques_dict_list_save, ques_dict_gen, dict_list_save
from tools.prompt_temp import get_react_messages
from langchain_community.utilities import GoogleSerperAPIWrapper

from tools.query_agents import get_answer_by_rag, get_answer_by_google

logging.basicConfig(filename='error-2.log', encoding='utf-8', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

os.environ["SERPER_API_KEY"] = ""
API_KEY = ''
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

llm = ChatZhipuAI(api_key=API_KEY, model='glm-4', max_tokens=8192, top_p=0.5, timeout=240, stream=False)

search = GoogleSerperAPIWrapper()


def simple_chat(input_content1):
    '''
    系统角色-金庸小说专家
    :param input_content1: 提问内容
    :return: 回答
    '''
    res = llm.invoke(
        [
            SystemMessage(content="你现在是一个金庸小说专家，解答各种与金庸小说相关的问题."),
            HumanMessage(content=input_content1)
        ]
    )

    return res


def get_ques_by_type(ques_type, novel_name=Jinyong.YiTian.value):
    messages = [
        SystemMessage(f"你是一个金庸小说《{novel_name}》专家,请你从以下九种小说问题类型中，围绕“{ques_type}”这种类型提出相关的问题。\n1"
                      f".人物行为动机2.人物关系3.武功秘籍4.角色介绍5.角色门派分类6.角色立场的好坏7.小说的重要事件8.小说时间线相关的问题9.人物角色故事"),
        AIMessage(f"问题：郭襄和张君宝是如何相识的？\n类型：人物关系"),
        AIMessage(f"问题：《倚天屠龙记》中，九阳神功是一门怎样的武功\n类型：武功秘籍"),
        AIMessage(f"问题：《倚天屠龙记》中，杨不悔的结局是怎样的？\n类型：人物角色故事"),
        HumanMessage(f"问题：")
    ]
    return llm.invoke(messages).content



def ques_gen_by_glm(question_count=0):
    completed_novels = []  # 已完成列表
    for novel_name in Jinyong:
        if novel_name in completed_novels:
            print(f'小说 {novel_name.value} 已处理过，跳过。')
            continue
        print('当前处理小说:', novel_name.value)

        while question_count < 300:
            ques_type = random.choice(list(QuestionType))
            question_count += 1
            dict_list = []
            try:
                ques_with_type = get_ques_by_type(ques_type, novel_name.value)
                dict_list.append(ques_dict_gen(ques_with_type, novel_name, ))
                ques_dict_list_save(dict_list)
                print(question_count, dict_list[0])
            except Exception as e:
                print('异常问题位置:', question_count)
                # 发生异常时，将异常信息保存到日志文件
                logging.error(f"基本信息:小说名称{novel_name.value}")
                logging.error(f"异常文档位置:{question_count}")
                logging.error(f"Exception occurred: {e}")
        print(f"{novel_name.value}处理完成!")
        question_count = 0  # 完成置0
        completed_novels.append(novel_name)
        print('当前完成列表', completed_novels)



def ques_gen_by_react(processed_count=0):
    # 打开 react_question.jsonl 文件
    with open('./data/react_question.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 循环处理每一行数据
    for i, line in enumerate(lines):
        if i < processed_count:
            continue
        print('当前处理问题位置：', processed_count)
        data = json.loads(line)
        question = data["question"]
        novel_name = data["novel_name"]

        messages = get_react_messages(novel_name, question)

        sub_questions = llm.invoke(messages).content
        # sub_questions = '1. 慕容复是金庸哪部小说中的角色？\n'
        dict_list = []
        for sub_question in sub_questions.split('\n'):
            print(sub_question)
            # 随机选择一种方法获取答案
            # methods = [get_answer_by_google, get_answer_by_rag, get_answer_by_wiki]
            methods = {
                get_answer_by_google: "Google", #欠费了,
                # get_answer_by_wiki:"Wiki",
                get_answer_by_rag: "RAG"
            }
            try:
                chosen_method = random.choice(list(methods.keys()))
                method_name = methods[chosen_method]
                answer = chosen_method(sub_question,novel_name)
                # 整理成字典并打印
                result = {
                    "question": sub_question,
                    "answer": answer,
                    "novel_name": novel_name,
                    "data_source": method_name
                }
                dict_list.append(result)
            except Exception as e:
                print('异常问题位置:', processed_count)
                # 发生异常时，将异常信息保存到日志文件
                logging.error(f"异常问题位置:{processed_count}")
                logging.error(f"Exception occurred: {e}")
        dict_list_save(dict_list)
        processed_count += 1
        print(dict_list)



if __name__ == '__main__':
    ques_gen_by_react(processed_count=130)
    # 复杂问题->拆解->‘
