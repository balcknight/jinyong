# coding=utf-8
from zhipuai import ZhipuAI

from langchain_glm.ZhipuChat import ChatZhipuAI
from messages.novel_enum import Jinyong, DataSource
from tools.text_load import text_split
from tools.db_load import load_retriever, get_rag_text_by_retriever
from tools.data_handle import dict_gen_by_novel, dict_list_save
import logging
API_KEY = ''
client = ZhipuAI(api_key="", timeout=240)  # 填写您自己的APIKey
zhipuai_chat = ChatZhipuAI(api_key=API_KEY,model='glm-4',max_tokens=8192,top_p=0.5,timeout=240,stream=False)
# documents = text_split(f'E:/workspace/cv_study/torch_study/base_usage/llm_study/jinyong/data/original_texts/{Jinyong.XueShan.value}.txt', chunk_size=4000)


# 设置日志记录器
logging.basicConfig(filename='error-2.log', encoding='utf-8', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# 加载向量数据库
def get_text_embedding(source_text):
    '''
    :param source_text: 输入的文本
    :return: 返回1024维度的列表向量
    '''
    response = client.embeddings.create(
        model="embedding-2",  # 填写需要调用的模型名称
        input=source_text,
    )
    return response.data[0].embedding


def get_ques_answer_by_source(source_text, novel_name=Jinyong.YiTian.value):
    messages = [
        {"role": "system", "content": f"你是一个金庸小说《{novel_name}》专家"},
        {"role": "user",
         "content": f"，请根据给出的《{novel_name}》小说原文，提出问题并进行回答。注意：提出的问题必须是从《{novel_name}》小说全局的角度提出问题（即不使用'这次事件'、'那个男人'、'谁谁谁在何处'等这样的局部事件描述词），且不能局限于给出的原文片段，提出的问题-答案尽可能详细，记住，不是对文章进行摘要总结（即不使用'这段原文描述了'、'这段原文是关于'、'这段文字'等等的回复），而是提出问题并回答"},
        {"role": "user", "content": """原文片段：\n
    第一回 风雪惊变
            两株大松树下围着一堆村民，男男女女和十几个小孩，正自聚精会神的听着一个瘦削的老者说话。
    　　那说话人五十来岁年纪，一件青布长袍早洗得褪成了蓝灰色。只听他两片梨花木板碰了几下，左手中竹棒在一面小羯鼓上敲起得得连声。唱道：
    　　“小桃无主自开花，烟草茫茫带晚鸦。
    　　几处败垣围故井，向来一一是人家。”
    　　那说话人将木板敲了几下，说道：“这首七言诗，说的是兵火过后，原来的家家户户，都变成了断墙残瓦的破败之地。小人刚才说到那叶老汉一家四口，悲欢离合，聚了又散，散了又聚。他四人给金兵冲散，好容易又再团聚，欢天喜地的回到故乡，却见房屋已给金兵烧得干干净净，无可奈何，只得去到汴梁，想觅个生计。不料想：天有不测风云，人有旦夕祸福。他四人刚进汴梁城，迎面便过来一队金兵。带兵的头儿一双三角眼觑将过去，见那叶三姐生得美貌，跳下马来，当即一把抱住，哈哈大笑，便将她放上了马鞍，说道：‘小姑娘，跟我回家，服侍老爷。’那叶三姐如何肯从？拚命挣扎。那金兵长官喝道：‘你不肯从我，便杀了你的父母兄弟！’提起狼牙棒，一棒打在那叶四郎的头上，登时脑浆迸裂，一命呜呼。正是：
    　　阴世新添枉死鬼，阳间不见少年人！
    　　“叶老汉和妈妈吓得呆了，扑将上去，搂住了儿子的死尸，放声大哭。那长官提起狼牙棒，一棒一个，又都了帐。那叶三姐却不啼哭，说道：‘长官休得凶恶，我跟你回家便了！’那长官大喜，将叶三姐带得回家。不料叶三姐觑他不防，突然抢步过去，拔出那长官的腰刀，对准了他心口，一刀刺将过去，说时迟，那时快，这一刀刺去，眼见便可报得父母兄弟的大仇。不料那长官久经战阵，武艺精熟，顺手一推，叶三姐登时摔了出去。那长官刚骂得一声：‘小贱人！’叶三姐已举起钢刀，在脖子中一勒。可怜她：
    　　花容月貌无双女，惆怅芳魂赴九泉。”
    　　他说一段，唱一段，只听得众村民无不咬牙切齿，愤怒叹息。
    　　那人又道：“众位看官，常言道得好：
    　　为人切莫用欺心，举头三尺有神明。
    　　若还作恶无报应，天下凶徒人吃人。
    　　“可是那金兵占了我大宋天下，杀人放火，奸淫掳掠，无恶不作，却又不见他遭到什么报应。只怪我大宋官家不争气，我中国本来兵多将广，可是一见到金兵到来，便远远的逃之夭夭，只剩下老百姓遭殃。好似那叶三姐一家的惨祸，江北之地，实是成千成万，便如家常便饭一般。诸君住在江南，当真是在天堂里了，怕只怕金兵何日到来。正是：宁作太平犬，莫为乱世人。小人张十五，今日路经贵地，服侍众位看官这一段说话，叫作《叶三姐节烈记》，话本说彻，权作散场。”将两片梨花木板拍拍拍的乱敲一阵，托出一只盘子。
    　　众村民便有人拿出两文三文，放入木盘，霎时间得了六七十文。张十五谢了，将铜钱放入囊中，便欲起行。
    　　村民中走出一个二十来岁的大汉，说道：“张先生，你可是从北方来吗？”张十五见他身材魁梧，浓眉大眼，便道：“正是。”那大汉道：“小弟作东，请先生去饮上三杯如何？”张十五大喜，说道：“素不相识，怎敢叨扰？”那大汉笑道：“喝上三杯，那便相识了。我姓郭，名叫郭啸天。”指着身旁一个白净面皮的汉子道：“这位是杨铁心杨兄弟。适才我二人听先生说唱叶三姐节烈记，果然是说得好，却有几句话想要请问。”张十五道：“好说，好说。今日得遇郭杨二位，也是有缘。”
    　　郭啸天带着张十五来到村头一家小酒店中，在张饭桌旁坐了。"""},
        {"role": "assistant", "content": """问题：郭啸天和张十五是如何认识的？\n回答：郭啸天和张十五是在村子里听张十五说唱《叶三姐节烈记》时认识的。张十五是一个说书人，正在用木板和羯鼓表演说唱《叶三姐节烈记》，引起了村民们的关注。郭啸天和杨铁心听到后觉得张十五说得好，于是邀请他来喝酒，并提出询问一些问题。这次邀请使得郭啸天和张十五结识
    """},
        {"role": "user", "content": "原文片段：\n" + source_text}
    ]
    response = client.chat.completions.create(
        model="glm-4",  # 填写需要调用的模型名称 glm-3-turbo glm-4
        max_tokens=8192,
        top_p=0.5,
        messages=messages
    ).choices[0].message.content
    return response

def data_gen_by_novel(chunk_size=4000, chunk_overlap=2350, processed_count=0):
    completed_novels = [Jinyong.YiTian, Jinyong.ShuJian, Jinyong.XiaKe,Jinyong.TianLong,Jinyong.BaiMa,Jinyong.SheDiao,Jinyong.BiXue,Jinyong.ShenDiao,Jinyong.XiaoAo,Jinyong.LianCheng,Jinyong.XueShan,Jinyong.YueNv,Jinyong.YuanYan,Jinyong.FeiHu]  # 已完成列表
    for novel_name in Jinyong:
        if novel_name in completed_novels:
            print(f'小说 {novel_name.value} 已处理过，跳过。')
            continue
        print('当前处理小说:', novel_name.value)
        documents = text_split(
            f'E:/workspace/cv_study/torch_study/base_usage/llm_study/jinyong/data/original_texts/{novel_name.value}.txt',
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print('小说片段数量:', len(documents))

        # 遍历文档，对每个文档调用 get_ques_answer_by_source，得到问题-答案列表

        for i, document in enumerate(documents):
            if i < processed_count:
                continue
            print('当前处理片段位置：', processed_count)
            novel_source = document.page_content
            # print('原文片段：\n', novel_source)
            dict_list = []
            try:
                ques_answers = get_ques_answer_by_source(novel_source, novel_name.value)
                dict_list.append(dict_gen_by_novel(ques_answers, novel_name, DataSource.NOVEL))
                dict_list_save(dict_list)
            except Exception as e:
                print('异常文档位置:', processed_count)
                # 发生异常时，将异常信息保存到日志文件
                logging.error(f"基本信息:小说名称{novel_name.value}，文档块大小{chunk_size}，重复块大小：{chunk_overlap}")
                logging.error(f"异常文档位置:{processed_count}")
                logging.error(f"Exception occurred: {e}")
            processed_count += 1
        processed_count = 0  # 处理完整个文档，下本小说置0
        print(f"{novel_name.value}处理完成!")
        completed_novels.append(novel_name)
        print('当前完成列表', completed_novels)


if __name__ == '__main__':
    # documents = text_split(f'E:/workspace/cv_study/torch_study/base_usage/llm_study/jinyong/data/original_texts/{Jinyong.SheDiao.value}.txt', chunk_size=4000,chunk_overlap=1300)
    # print(documents[251].page_content)
    # print(get_ques_answer_by_source(documents[82].page_content, Jinyong.XiaKe))
    data_gen_by_novel(chunk_size=4000, chunk_overlap=1300, processed_count=193)


# print(get_answer_by_rag("为什么谢逊要离开张翠山夫妇和无忌？"))
# print(data_gen_by_novel())
# print(get_ques_answer_by_source(documents[5].page_content,Jinyong.XueShan.value))
