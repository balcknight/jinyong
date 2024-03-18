import time
import jwt
from langchain import hub

from llm_study.jinyong.langchain_glm.ZhipuChat import ChatZhipuAI

apikey = ''
token = ""
exp_seconds = 360000

API_KEY = ''


def contention_test():
    zhipuai_chat = ChatZhipuAI(
        api_key=API_KEY,
        model='glm-4',
        max_tokens=8192,
        top_p=0.5,
        timeout=240,
        stream=False
    )

    print(zhipuai_chat.invoke("你是谁哪个版本的glm？").content)


def generate_token(apikey: str, exp_seconds: int):
    '''
    获取令牌
    :param apikey:
    :param exp_seconds: 令牌过期时间
    :return:
    '''
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }

    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )

# print(generate_token(apikey,exp_seconds))
