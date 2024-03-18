import os

from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentType, Tool, initialize_agent

os.environ['OPENAI_API_KEY'] = ''

os.environ["SERPER_API_KEY"] = ""
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import GoogleSerperAPIWrapper

llm = OpenAI(top_p=0.9, max_tokens=3000)
search = GoogleSerperAPIWrapper(lang='zh')


def wikiApi_ues(query_key):
    # 工具包含 name describe 输入内容的json格式， 调用函数 结果是否直接返回给用户
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, lang='zh', doc_content_chars_max=10000)
    tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    print(tool.name)
    print(tool.description)
    print(tool.args)
    print(tool.return_direct)
    res = tool.run(query_key)
    print(res)  # 调用函数
    return res


def simple_chat(input_content1):
    res = llm.invoke(
        [
            SystemMessage(content="你现在是一个金庸小说专家，解答各种与金庸小说相关的问题."),
            HumanMessage(content=input_content1)
        ]
    )

    return res


# print(simple_chat('你好'))
search = GoogleSerperAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]

self_ask_with_search = initialize_agent(
    tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
)
self_ask_with_search.run(
    "北京的天气是什么？"
)

# print()


# 复杂问题->拆解->‘

"""
你是金庸小说专家，
金庸所有的小说被存储在向量数据库，请将以下关于金庸小说的问题生成三个相关且更简单的小问题或者关键词，用于查询向量数据库匹配相关的内容：
请注意，每个问题的分解应确保相关性，并使每个问题的解决过程逐步、清晰。


金庸小说中，杨不悔的结局是什么？
1.杨不悔是哪部金庸小说中的角色？ 
2.杨不悔的角色背景和经历是什么？
3.杨不悔最终的归宿和情感结局是怎样的？

 虚竹的主要关系网络？这些关系如何影响他的命运？
1.虚竹是哪部金庸小说中的角色？
2.虚竹在小说中有什么特殊的身份或地位？
3.虚竹的主要关系网包括哪些人物？

杨过和小龙女之间的感情纠葛以及最终的故事结局是什么？
1.杨过和小龙女分别是哪部金庸小说中的角色？
2.杨过和小龙女之间的感情纠葛经历了哪些波折？
3.杨过和小龙女最终的故事结局是怎样的？
"""
