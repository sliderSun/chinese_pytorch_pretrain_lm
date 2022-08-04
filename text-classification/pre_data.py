"""
@Time : 2022/8/3 15:48 
@Author : sunshb10145 
@File : pre_data.py 
@desc:
"""
import fairies as fa
from tqdm import tqdm
import csv
# data_file = open("./data/cf/valid_cf.csv", "w", encoding="utf-8", newline="")
# csv_writer = csv.writer(data_file)
# csv_writer.writerow(["sentence1", "sentence2", "label"])
# data = fa.read_json("./data/cf/valid_cf.json")
# for d in tqdm(data):
#     print([d[0], d[1], d[2]])
#     csv_writer.writerow([d[0], d[1], d[2]])
# data_file.close()
data_names = ["test", "valid", "train"]
for data in data_names:
    with open("./data/LCQMC/LCQMC.{}.data".format(data), "r", encoding="utf-8") as fr:
        with open("./data/LCQMC/{}.csv".format(data), "w", encoding="utf-8", newline="") as fw:
            csv_writer = csv.writer(fw)
            csv_writer.writerow(["sentence1", "sentence2", "label"])
            lines = fr.readlines()
            for line in lines:
                line = line.replace("\n", "")
                s1, s2, label = line.split("\t")
                csv_writer.writerow([s1, s2, int(label)])



