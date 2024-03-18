import os

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents import tool, create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate

from langchain_glm.ZhipuChat import ChatZhipuAI




def get_react_messages(novel_name, user_ques):
    messages = [
        SystemMessage(
            f"你是一个金庸小说《{novel_name}》专家，请将以下关于金庸小说《{novel_name}》的问题拆解成三个相关且更简单的小问题或者关键词，用于查询相关的内容，请注意，只允许拆解成3个问题，每个问题的分解应确保相关性，并使每个问题的解决过程逐步、清晰。"),
        HumanMessage("问题：金庸小说中，杨不悔的结局是什么？"),
        AIMessage("1.杨不悔是金庸哪部小说中的角色？\n2.杨不悔的角色背景和经历是什么？\n3.杨不悔最终的归宿和情感结局是怎样的？\n"),
        HumanMessage("虚竹的主要关系网络？这些关系如何影响他的命运？"),
        AIMessage("1.虚竹是哪部金庸小说中的角色？\n2.虚竹在小说中背景和经历是什么？\n3.虚竹的主要关系网包括哪些人物？\n"),
        HumanMessage("问题：杨过和小龙女之间的感情纠葛以及最终的故事结局是什么？"),
        AIMessage("1.杨过和小龙女分别是金庸哪部小说中的角色？\n2.杨过和小龙女之间的感情纠葛经历了哪些波折？\n3.杨过和小龙女最终的故事结局是怎样的？\n"),
        HumanMessage("问题：张翠山和殷姑娘是如何认识的？"),
        AIMessage("1.张翠山和殷姑娘分别是金庸哪部小说中的角色？ \n2.殷姑娘在小说中的身份和背景是什么？ \n3.张翠山和殷姑娘是如何相遇并建立关系的？\n"),
        HumanMessage("问题：" + user_ques)
    ]
    return messages


def get_react_answer_messages(user_ques):
    messages = [
        SystemMessage(
            f"你是一个金庸小说专家，请将以下关于金庸小说的问题拆解成三个相关且更简单的小问题或者关键词，用于查询相关的内容，请注意，只需要输出3个问题，每个问题的分解应确保相关性，并使每个问题的解决过程逐步、清晰。"),
        HumanMessage("问题：金庸小说中，杨不悔的结局是什么？"),
        AIMessage("1.杨不悔是金庸哪部小说中的角色？\n2.杨不悔的角色背景和经历是什么？\n3.杨不悔最终的归宿和情感结局是怎样的？\n"),
        HumanMessage("虚竹的主要关系网络？这些关系如何影响他的命运？"),
        AIMessage("1.虚竹是哪部金庸小说中的角色？\n2.虚竹在小说中背景和经历是什么？\n3.虚竹的主要关系网包括哪些人物？\n"),
        HumanMessage("问题：杨过和小龙女之间的感情纠葛以及最终的故事结局是什么？"),
        AIMessage("1.杨过和小龙女分别是金庸哪部小说中的角色？\n2.杨过和小龙女之间的感情纠葛经历了哪些波折？\n3.杨过和小龙女最终的故事结局是怎样的？\n"),
        HumanMessage("问题：张翠山和殷姑娘是如何认识的？"),
        AIMessage("1.张翠山和殷姑娘分别是金庸哪部小说中的角色？ \n2.殷姑娘在小说中的身份和背景是什么？ \n3.张翠山和殷姑娘是如何相遇并建立关系的？\n"),
        HumanMessage("问题：" + user_ques)
    ]
    return messages


def get_rag_messages(novel_name, user_ques, rag_content):
    '''
    获取rag的消息模板
    :param novel_name: 小说名称
    :param user_ques: 用户问题
    :param rag_content: 检索内容
    :return:
    '''
    messages = [
        SystemMessage(
            f"你是一个金庸小说《{novel_name}》专家，根据提供的信息来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分。"),
        HumanMessage(f"已知信息：{rag_content} \n 问题：{user_ques}")
    ]
    return messages


def get_rag_messages( user_ques, rag_content):
    '''
    获取rag的消息模板
    :param user_ques: 用户问题
    :param rag_content: 检索内容
    :return:
    '''
    messages = [
        SystemMessage(
            f"你是一个金庸小说专家，根据提供的信息来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分。"),
        HumanMessage(f"已知信息：{rag_content} \n 问题：{user_ques}")
    ]
    return messages
