import os

from langchain_core.utils.function_calling import format_tool_to_openai_tool
from langchain_openai import OpenAI,ChatOpenAI

from langchain_glm.ZhipuChat import ChatZhipuAI
from langchain.agents import tool, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
os.environ['OPENAI_API_KEY']= ''

os.environ["SERPER_API_KEY"] = ""
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
API_KEY = ''

llm = ChatZhipuAI(
    api_key=API_KEY,
    model='glm-4',
    max_tokens=8192,
    top_p=0.5,
    timeout=240,
    stream=False
)
# llm = ChatOpenAI(top_p=0.9,max_tokens=3000)

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)


# print(get_word_length.invoke("abc"))

tools = [get_word_length]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", "You are very powerful assistant, but don't know current events",
        ),
        ("user", "{input}"),

        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# print(prompt.format(input="How many letters in the word eudcasssssssssssss", agent_scratchpad=['get_word_length']))

agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
print(list(agent_executor.stream({"input": "How many letters in the word eudcasssssssssssss"})))