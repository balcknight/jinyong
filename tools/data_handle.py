# -*- coding: utf-8 -*-
import random
import string

from messages.novel_enum import Jinyong, DataSource
import jsonlines

import json

def trans_dict_to_dataset(data_path='converted_dataset.json',out_path='data/dataset.jsonl'):
    '''
    字典数据转对话格式
    :param data_path:字典数据
    :param out_path:对话格式数据路径
    :return:
    '''
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    output_data = []
    for item in data:
        conversations = [
            {"from": "human", "value": item["conversations"]["question"]},
            {"from": "gpt", "value": item["conversations"]["answer"]}
        ]
        output_item = {
            "conversations": conversations,
            "novel_name": item["novel_name"],
            "data_source": item["data_source"]
        }
        output_data.append(output_item)

    with open(out_path, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, ensure_ascii=False, indent=2)


def ques_dict_list_save(dicts,out_path='data/dataset.jsonl'):
    # 将字典列表逐行写入到 dataset.jsonl 文件中
    with jsonlines.open(out_path, mode='a') as writer:
        for d in dicts:
            writer.write(d)


def dict_gen_by_novel(string="aaa", novel_name=Jinyong.YiTian, data_source=DataSource.RAG):
    is_classified = False
    # 检查字符串中是否存在 "问题：" 和 "回答："，如果存在则以这两个切割，否则以"问题"、"回答"进行切割
    if "问题：" in string and "回答：" in string:
        question, answer = string.split("问题：")[1].split("回答：")
    elif "问题" in string and "回答" in string:
        question, answer = string.split("问题")[1].split("回答")
    elif "问题" in string and "答案" in string:
        question, answer = string.split("问题")[1].split("答案")
    else:
        # 如果既没有 "问题" 也没有 "回答"，则抛出异常并打印当前字符串
        raise ValueError("无法找到问题和回答:", string)
    # 去除可能存在的换行符和空格
    question = question.strip()
    answer = answer.strip()
    # 创建字典并添加到字典列表中
    dict = {"question": question, "answer": answer, "novel_name": novel_name.value, "data_source": data_source.value,
            "is_classified": is_classified}
    return dict


def ques_dict_gen(input_string,novel_name=Jinyong.YiTian):
    '''
    解析llm给出的问题数据为字典
    :param input_string:
    :param novel_name:
    :return:
    '''
    result = {}
    lines = input_string.split('\n')
    result['question'] = lines[0]
    if "类型：" in input_string:
        result['ques_type'] = input_string.split("类型：")[1]
    elif "类型" in input_string:
        result['ques_type'] = input_string.split("类型")[1]
    else:
        # 如果既没有 "问题" 也没有 "回答"，则抛出异常并打印当前字符串
        raise ValueError("无法找到问题或问题类型:", input_string)
    result['novel_name']=novel_name.value
    return result

def random_dataset(dataset_path='dataset.jsonl'):
    # 打开原始数据文件
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = f.readlines()
    # 从原始数据中随机选择500条
    random_selection = random.sample(dataset, 500)

    # 将选定的数据写入新文件
    with open('react_question.jsonl', 'w', encoding='utf-8') as f:
        for line in random_selection:
            f.write(line)

    print("数据已写入到 react_question.jsonl 文件中。")


if __name__ == '__main__':
    random_dataset('../data/dataset.jsonl')
