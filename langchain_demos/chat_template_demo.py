# -*- coding:utf-8 -*-
# Chat Messages
# 三种类型
# System - 指挥人家要做啥，完成一件什么事
# Human - 我的描述和问题事什么
# AI - 人家LLM回答我的问题
import os

from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
OPENAI_API_KEY = ''
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate,HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory
# proxy='127.0.0.1:7890'
chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=.7, max_tokens=4096, model="gpt-3.5-turbo",
                  request_timeout=20)
# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) # 向量化编码很重要，查和匹配都会用得上

# chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

# 可以链式调用invoke函数
# parser = StrOutputParser()


def simple_chat(input_content):
    res = chat.invoke(
        [
            SystemMessage(content="你现在是一个软件产品经理，请根据给出的软件需求生成软件需求文档提纲."),
            HumanMessage(content=input_content),
            # AIMessage(content="我们必须尊重媳妇的意见"),
            # HumanMessage(content="对输入的文章生成3条主要摘要,分别从故事的起因，发展，结束三个角度进行概述")
            # HumanMessage(content="分别从故事的起因，发展，结束三个角度进行概述")
        ]
    ).content
    return res

def chat_with_history(historys,system_mess="你现在是一个软件产品经理，请根据给出的软件需求生成软件需求文档提纲."):
    history = ChatMessageHistory()
    history.add_message(SystemMessage(content=system_mess)) # 添加系统角色信息
    history.messages+=historys # 添加历史对话信息

    res = chat.invoke(
        history.messages
    ).content
    history.add_ai_message(res)
    return history.messages

def template_simple_use():
    template1 = """
    根据人名来起一个外号.
    这个名字：{product}，应该起什么外号会更有意思?
    """
    prompt = PromptTemplate.from_template(template1)
    print(prompt)
    print(prompt.format(product="刘德华"))

def template_standard_use():
    template2="请将输入的{input_language}句子翻译成 {output_language}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template2)
    human_template="输入:{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt]) # 组合信息
    chat_prompt=chat_prompt.format_prompt(input_language="中文", output_language="英文", text="我快累死了.").to_messages()
    return chat_prompt

def text_split():
    document = """
    # 食堂饭卡管理系统软件需求文档提纲\n\n## 1. 引言\n### 1.1 目的\n### 1.2 范围\n### 1.3 定义\n\n## 2. 产品描述\n### 2.1 产品概述\n### 2.2 产品功能\n### 2.3 用户角色\n\n## 3. 需求规定\n### 3.1 功能性需求\n#### 3.1.1 用户注册与登录\n#### 3.1.2 饭卡充值\n#### 3.1.3 饭卡消费\n#### 3.1.4 饭卡余额查询\n#### 3.1.5 消费记录查询\n#### 3.1.6 数据统计与分析\n\n### 3.2 非功能性需求\n#### 3.2.1 安全性要求\n#### 3.2.2 易用性要求\n#### 3.2.3 可靠性要求\n#### 3.2.4 性能要求\n\n## 4. 界面设计\n### 4.1 登录界面\n### 4.2 充值界面\n### 4.3 消费界面\n### 4.4 查询界面\n### 4.5 统计界面\n\n## 5. 数据库设计\n### 5.1 用户信息表\n### 5.2 饭卡信息表\n### 5.3 消费记录表\n\n## 6. 系统架构\n### 6.1 客户端架构\n### 6.2 服务器架构\n### 6.3 数据库架构\n\n## 7. 部署与实施\n### 7.1 环境要求\n### 7.2 部署步骤\n### 7.3 测试计划\n\n## 8. 维护与支持\n### 8.1 常见问题解答\n### 8.2 紧急支持联系方式\n\n## 9. 附录\n### 9.1 术语表\n### 9.2 参考文献\n\n以上是软件需求文档提纲的初步草拟，具体内容仍需根据实际情况进行进一步补充和细化。
    """
    sections = document.split("\n\n## ")[1:]
    document_list=[]
    for section in sections:
        section_text=''
        section_parts = section.split("\n### ")
        section_title = section_parts[0]
        subsections = section_parts[1:]
        section_text=section_title
        for subsection in subsections:
            section_text+=subsection
        print(section_text)


# text_split()
# history=chat_with_history("""
#             食堂饭卡管理系统是一个集充值、消费、查询等功能于一体的信息化系统，旨在提高食堂管理效率，方便师生就餐。
# 请你根据以上需求生成一份软件文档提纲
#             """)
history=[HumanMessage(content="""
            食堂饭卡管理系统是一个集充值、消费、查询等功能于一体的信息化系统，旨在提高食堂管理效率，方便师生就餐。
请你根据以上需求生成一份软件文档提纲
            """)]

history.append(AIMessage(content="""# 食堂饭卡管理系统软件需求文档提纲\n\n## 1. 引言\n### 1.1 目的\n### 1.2 范围\n### 1.3 定义\n\n## 2. 产品描述\n### 2.1 产品概述\n### 2.2 产品功能\n### 2.3 用户角色\n\n## 3. 需求规定\n### 3.1 功能性需求\n#### 3.1.1 用户注册与登录\n#### 3.1.2 饭卡充值\n#### 3.1.3 饭卡消费\n#### 3.1.4 饭卡余额查询\n#### 3.1.5 消费记录查询\n#### 3.1.6 数据统计与分析\n\n### 3.2 非功能性需求\n#### 3.2.1 安全性要求\n#### 3.2.2 易用性要求\n#### 3.2.3 可靠性要求\n#### 3.2.4 性能要求\n\n## 4. 界面设计\n### 4.1 登录界面\n### 4.2 充值界面\n### 4.3 消费界面\n### 4.4 查询界面\n### 4.5 统计界面\n\n## 5. 数据库设计\n### 5.1 用户信息表\n### 5.2 饭卡信息表\n### 5.3 消费记录表\n\n## 6. 系统架构\n### 6.1 客户端架构\n### 6.2 服务器架构\n### 6.3 数据库架构\n\n## 7. 部署与实施\n### 7.1 环境要求\n### 7.2 部署步骤\n### 7.3 测试计划\n\n## 8. 维护与支持\n### 8.1 常见问题解答\n### 8.2 紧急支持联系方式\n\n## 9. 附录\n### 9.1 术语表\n### 9.2 参考文献\n\n以上是软件需求文档提纲的初步草拟，具体内容仍需根据实际情况进行进一步补充和细化。"""))

history_sp=history.copy()

history_sp.append(HumanMessage(content="很好，请在每个部分中间增加分隔符"))

print(history)

summary='# 食堂饭卡管理系统软件需求文档提纲\n\n---\n\n## 1. 引言\n### 1.1 目的\n### 1.2 范围\n### 1.3 定义\n\n---\n\n## 2. 产品描述\n### 2.1 产品概述\n### 2.2 产品功能\n### 2.3 用户角色\n\n---\n\n## 3. 需求规定\n### 3.1 功能性需求\n#### 3.1.1 用户注册与登录\n#### 3.1.2 饭卡充值\n#### 3.1.3 饭卡消费\n#### 3.1.4 饭卡余额查询\n#### 3.1.5 消费记录查询\n#### 3.1.6 数据统计与分析\n\n### 3.2 非功能性需求\n#### 3.2.1 安全性要求\n#### 3.2.2 易用性要求\n#### 3.2.3 可靠性要求\n#### 3.2.4 性能要求\n\n---\n\n## 4. 界面设计\n### 4.1 登录界面\n### 4.2 充值界面\n### 4.3 消费界面\n### 4.4 查询界面\n### 4.5 统计界面\n\n---\n\n## 5. 数据库设计\n### 5.1 用户信息表\n### 5.2 饭卡信息表\n### 5.3 消费记录表\n\n---\n\n## 6. 系统架构\n### 6.1 客户端架构\n### 6.2 服务器架构\n### 6.3 数据库架构\n\n---\n\n## 7. 部署与实施\n### 7.1 环境要求\n### 7.2 部署步骤\n### 7.3 测试计划\n\n---\n\n## 8. 维护与支持\n### 8.1 常见问题解答\n### 8.2 紧急支持联系方式\n\n---\n\n## 9. 附录\n### 9.1 术语表\n### 9.2 参考文献\n\n---\n\n以上是软件需求文档提纲的初步草拟，具体内容仍需根据实际情况进行进一步补充和细化。'

results=summary.split('---')

results.pop(0) # 删除标题
results=results[:-1] # 删除结尾
output=''
for res in results:
    temp=history.copy()
    template = """
        请你给“#{document_sec}”部分进行扩写，生成相应的内容
        """
    prompt = PromptTemplate.from_template(template)

    temp.append(HumanMessage(content=prompt.format(document_sec=res)))
    res_f=chat_with_history(temp)
    output+=res_f[-1].content
    output +='\n'
    print(res_f[-1].content)

with open('output.txt', 'w') as file:  # 打开文件以写入模式
    file.write(output)
# print(chat_with_history(history))