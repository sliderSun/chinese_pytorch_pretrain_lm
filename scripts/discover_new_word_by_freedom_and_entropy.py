"""
@Time : 2022/7/24 21:29 
@Author : sunshb10145 
@File : discover_new_word_by_freedom_and_entropy.py 
@desc:
"""
'''
数据获取及预处理；词典获取
'''
# 读取数据
from tqdm import tqdm
import re
from ltp import LTP

ltp = LTP()


def preprocess_data(file_path):
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[:100000]
        with tqdm(lines, total=len(lines)) as pbar:
            for text in pbar:
                text = re.sub("[^\u4e00-\u9fa5。？．，！：]", "", text.strip())
                #             text_splited = re.split("[。？．，！：]", text)
                text_splited = ltp.sent_split([text])  # 调用LTP进行分句
                texts.extend(text_splited)

    tmp = texts
    texts = []
    with tqdm(tmp, total=len(tmp), desc="filtering the null sentences") as pbar:
        for text in pbar:
            if text is not "":
                texts.append(text)
    return texts


texts = preprocess_data("../data/test.txt")  # 按照基本的标点符号进行切分


# 获取已有的中文词典
def get_chinese_words(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.split()[0] for line in f.readlines()]


CH_DICT = set(get_chinese_words("../data/chinese_words.txt"))

'''
将数据进行切分获取所有切分出的候选单词，并且统计词频信息、候选新词左右出现的字的信息
接下来需要对文本进行切分以及获取相关的频次信息，这里统一在一个函数中，主要逻辑如下：
对文本按照一定的长度范围进行切分，切分出所有成词的可能性，这里称之为字符串；
对于所有切分出的字符串进行过滤，长度大于等于 2 的词以及不是词典 CH_DICT 中的词作为候选新词；
获取所有切分出的字符串的频次信息（在后续计算中需要用到一些字符串的频次信息）、候选新词词频信息、候选新词左右出现的字的统计信息。
'''


def get_candidate_wordsinfo(texts, max_word_len):
    '''
    texts：表示输入的所有文本
    max_word_len：表示最长的词长
    '''
    # 四个词典均以单词为 key，分别以词频、候选新词词频、左字集合、右字集合为 value
    words_freq, candidate_words_freq, candidate_words_left_characters, candidate_words_right_characters = {}, {}, {}, {}
    WORD_NUM = 0  # 统计所有可能的字符串频次
    with tqdm(texts, total=len(texts)) as pbar:
        for text in pbar:  # 遍历每个文本
            # word_indexes 中存储了所有可能的词汇的切分下标 (i,j) ，i 表示词汇的起始下标，j 表示结束下标，注意这里有包括了所有的字
            # word_indexes 的生成需要两层循环，第一层循环，遍历所有可能的起始下标 i；第二层循环，在给定 i 的情况下，遍历所有可能的结束下标 j
            word_indexes = [(i, j) for i in range(len(text)) for j in range(i + 1, i + 1 + max_word_len)]
            WORD_NUM += len(word_indexes)
            for index in word_indexes:  # 遍历所有词汇的下标
                word = text[index[0]:index[1]]  # 获取单词
                # 更新所有切分出的字符串的频次信息
                if word in words_freq:
                    words_freq[word] += 1
                else:
                    words_freq[word] = 1
                if len(word) >= 2 and word not in CH_DICT:  # 长度大于等于 2 的词以及不是词典中的词作为候选新词
                    # 更新候选新词词频
                    if word in candidate_words_freq:
                        candidate_words_freq[word] += 1
                    else:
                        candidate_words_freq[word] = 1
                    # 更新候选新词左字集合
                    if index[0] != 0:  # 当为文本中首个单词时无左字
                        if word in candidate_words_left_characters:
                            candidate_words_left_characters[word].append(text[index[0] - 1])
                        else:
                            candidate_words_left_characters[word] = [text[index[0] - 1]]
                    else:
                        if word in candidate_words_left_characters:
                            candidate_words_left_characters[word].append(len(candidate_words_left_characters[word]))
                        else:
                            candidate_words_left_characters[word] = [0]
                    # 更新候选新词右字集合
                    if index[1] < len(text) - 1:  # 当为文本中末个单词时无右字
                        if word in candidate_words_right_characters:
                            candidate_words_right_characters[word].append(text[index[1]])  #
                        else:
                            candidate_words_right_characters[word] = [text[index[1]]]
                    else:
                        if word in candidate_words_right_characters:
                            candidate_words_right_characters[word].append(len(candidate_words_right_characters[word]))
                        else:
                            candidate_words_right_characters[word] = [0]
    return WORD_NUM, words_freq, candidate_words_freq, candidate_words_left_characters, candidate_words_right_characters


WORD_NUM, words_freq, candidate_words_freq, candidate_words_left_characters, candidate_words_right_characters = \
    get_candidate_wordsinfo(texts=texts, max_word_len=6)  # 字符串最长为 3

'''
根据第二步中统计的进行 pmi 值以及左右邻字熵的计算
'''

import math


# 计算候选单词的 pmi 值
def compute_pmi(words_freq, candidate_words_freq):
    words_pmi = {}
    with tqdm(candidate_words_freq, total=len(candidate_words_freq), desc="Counting pmi") as pbar:
        for word in pbar:
            # 首先，将某个候选单词按照不同的切分位置切分成两项，比如“电影院”可切分为“电”和“影院”以及“电影”和“院”
            bi_grams = [(word[0:i], word[i:]) for i in range(1, len(word))]
            # 对所有切分情况计算 pmi 值，取最大值作为当前候选词的最终 pmi 值
            # words_freq[bi_gram[0]]，words_freq[bi_gram[1]] 分别表示一个候选新词的前后两部分的出现频次
            words_pmi[word] = max(map(lambda bi_gram: math.log( \
                words_freq[word] / (words_freq[bi_gram[0]] * words_freq[bi_gram[1]] / WORD_NUM)), bi_grams))
    return words_pmi


'''
在下一步中，计算 pmi 值以及左右邻字熵。
'''
words_pmi = compute_pmi(words_freq, candidate_words_freq)

from collections import Counter


# 计算候选单词的邻字熵
def compute_entropy(candidate_words_characters):
    words_entropy = {}
    with tqdm(candidate_words_characters.items(), total=len(candidate_words_characters),
              desc="Counting entropy") as pbar:
        for word, characters in pbar:
            character_freq = Counter(characters)  # 统计邻字的出现分布
            # 根据出现分布计算邻字熵
            words_entropy[word] = sum(
                map(lambda x: - x / len(characters) * math.log(x / len(characters)), character_freq.values()))
    return words_entropy


words_left_entropy = compute_entropy(candidate_words_left_characters)
words_right_entropy = compute_entropy(candidate_words_right_characters)

'''
设定各指标的阈值，根据其值获取最终的新词结果
'''


# 根据各指标阈值获取最终的新词结果
def get_newwords(candidate_words_freq,
                 words_pmi,
                 words_left_entropy,
                 words_right_entropy,
                 words_freq_limit=4,
                 pmi_limit=5,
                 entropy_limit=1):
    # 在每一项指标中根据阈值进行筛选
    candidate_words = [k for k, v in candidate_words_freq.items() if v >= words_freq_limit]
    candidate_words_pmi = [k for k, v in words_pmi.items() if v >= pmi_limit]
    candidate_words_left_entropy = [k for k, v in words_left_entropy.items() if v >= entropy_limit]
    candidate_words_right_entropy = [k for k, v in words_right_entropy.items() if v >= entropy_limit]
    # 对筛选结果进行合并
    return list(set(candidate_words).intersection(candidate_words_pmi, candidate_words_left_entropy,
                                                  candidate_words_right_entropy))


# 可以不断调参数来达到想要的结果
new_words = get_newwords(candidate_words_freq,
                         words_pmi,
                         words_left_entropy,
                         words_right_entropy,
                         words_freq_limit=2,
                         pmi_limit=3,
                         entropy_limit=1)
print(len(new_words))
'''
过滤掉一些不正确的新词
'''
new_words1 = list(filter(lambda x: not re.search("[^\u4e00-\u9fa5]", x), new_words))
new_words2 = list(filter(lambda x: not re.search("[了但里的和为是]", x), new_words1))

print(len(new_words2))

with open("../data/new_words.txt", "w", encoding="utf-8") as f:
    for new_word in new_words2:
        f.write(new_word + "\n")
