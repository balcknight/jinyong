from enum import Enum

class Jinyong(Enum):
    YiTian = "倚天屠龙记"
    ShuJian = "书剑恩仇录"
    XiaKe = "侠客行"
    TianLong = "天龙八部"
    SheDiao = "射雕英雄传"
    BaiMa = "白马啸西风"
    BiXue = "碧血剑"
    ShenDiao = "神雕侠侣"
    XiaoAo = "笑傲江湖"
    YueNv = "越女剑"
    LianCheng = "连城诀"
    XueShan = "雪山飞狐"
    FeiHu = "飞狐外传"
    YuanYan = "鸳鸯刀"
    LvDing = "鹿鼎记"


class QuestionType(Enum):
    CharacterMotivation = 0 # 人物行为动机
    CharacterRelationship = 1 # 人物关系
    MartialArtsSecrets = 2 # 武功秘籍
    CharacterIntroduction = 3 # 角色介绍
    CharacterFactionClassification = 4 # 角色门派分类
    CharacterAlignment = 5 # 角色立场的好坏
    ImportantEvents = 6 # 小说的重要事件
    NovelTimeline = 7 # 小说时间线相关的问题
    CharacterStory = 8 # 人物角色故事


class DataSource(Enum):
    RAG = "rag"
    NOVEL = "novel"
    GOOGLE = "google"
    WIKI = "wiki"
