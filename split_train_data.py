"""
@Time : 2022/7/22 15:40 
@Author : sunshb10145 
@File : split_train_data.py 
@desc:

"""
import random

"""
随机按⽐例拆分数据
"""


def data_split(old_path, train_path, test_path, shuffle=False, ratio=0.8):
    all_list = []
    with open(old_path, "r", encoding="utf-8") as f:
        all_list.extend(f.readlines())
    num = len(all_list)
    if num >= (10 ** 5):
        ratio = 0.9
    elif num >= (10 ** 6):
        ratio = 0.95
    offset = int(num * ratio)
    if num == 0 or offset < 1:
        return [], all_list
    if shuffle:
        random.shuffle(all_list)  # 列表随机排序
    train = all_list[:offset]
    test = all_list[offset:]
    with open(train_path, "w", encoding="utf-8") as tr, open(test_path, "w", encoding="utf-8") as te:
        for d in train:
            tr.write(d)
        for e in test:
            te.write(e)
    return train, test


data_split("train_www.txt", "./data/train.txt", "./data/test.txt")
