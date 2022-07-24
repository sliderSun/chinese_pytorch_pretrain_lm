"""
@Time : 2022/7/24 21:21 
@Author : sunshb10145 
@File : discover_new_word_by_frequency.py 
@desc:
"""
from tqdm import tqdm
from ltp import LTP

ltp = LTP()
with open("../data/test.txt", "r", encoding="utf-8") as f:
    texts = f.readlines()[:1000]  # 因语料太大，所以这里只用了前1W条做新词发现
    with tqdm(range(0, len(texts), 100)) as pbar:
        words = []
        for i in pbar:
            words.extend([word for text in ltp.seg(texts[i:i + 100])[0] for word in text])
print(len(words))


def get_chinese_words(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.split()[0] for line in f.readlines()]


CH_DICT = set(get_chinese_words("../data/chinese_words.txt"))

import re

unigram_freq, bigram_freq = {}, {}
for i in range(len(words) - 1):
    if words[i] not in CH_DICT and not re.search("[^\u4e00-\u9fa5]", words[i]):
        if words[i] in unigram_freq:  # 一阶计数
            unigram_freq[words[i]] += 1
        else:
            unigram_freq[words[i]] = 1
    bigram = words[i] + words[i + 1]
    if bigram not in CH_DICT and not re.search("[^\u4e00-\u9fa5]", bigram):
        if bigram in bigram_freq:
            bigram_freq[bigram] += 1
        else:
            bigram_freq[bigram] = 1
unigram_freq_sorted = sorted(unigram_freq.items(), key=lambda d: d[1], reverse=True)
bigram_freq_sorted = sorted(bigram_freq.items(), key=lambda d: d[1], reverse=True)

print("unigram:\n", unigram_freq_sorted[:100])
print("\n")
print("bigram:\n", bigram_freq_sorted[:100])
