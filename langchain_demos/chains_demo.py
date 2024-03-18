import os

from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate

from langchain_glm.ZhipuChat import ChatZhipuAI

API_KEY = ''

chat = ChatZhipuAI(
    api_key=API_KEY,
    model='glm-4',
    max_tokens=8192,
    top_p=0.5,
    timeout=240
)

def summary_chain():
    # 自定义提示词模板
    prompt_template = """总结下文内容:
    {text}
    """
    # 生成提示词模板
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    loader = TextLoader('data/yitian.txt',encoding='UTF-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)  # 文档切割

    texts = text_splitter.split_documents(documents)
    print(len(texts))
    chain = load_summarize_chain(chat, chain_type="map_reduce", verbose=True,map_prompt=PROMPT,combine_prompt=PROMPT)

    # 直接拆分会导致超过限制，不能得到摘要
    res=chain.run(texts[200:205])
    print(res)

summary_chain()