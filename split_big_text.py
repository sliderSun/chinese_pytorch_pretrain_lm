"""
@Time : 2022/7/22 16:58 
@Author : sunshb10145 
@File : split_big_text.py 
@desc:
"""
import os

# 要分割的文件
source_file = 'D:\sunshubing\siwei\\new_data\训练数据content2\\train_www_data.txt'

# 定义每个子文件的行数
file_count = 20  # 根据需要自定义


def mk_SubFile(lines, srcName, sub):
    [des_filename, extname] = os.path.splitext(srcName)
    filename = des_filename + '_' + str(sub) + extname
    print('正在生成子文件: %s' % filename)
    with open(filename, 'wb') as fout:
        fout.writelines(lines)
        return sub + 1


def split_By_LineCount(filename, count):
    with open(filename, 'rb') as fin:
        buf = []
        sub = 1
        for line in fin:
            if len(line.strip()) > 0:  # 跳过空行
                buf.append(line)
                # 如果行数超过指定的数，且数据为一个完整的记录，则将buf写入到一个子文件中，并初始化buf
                line_tag = line.strip()[0]  # 取每一行第一个字符，如果该行为空，会报错,故加上前面判断
                if len(buf) >= count:  # 每一个新的记录数据是从*标识开始
                    buf = buf[:-1]
                    sub = mk_SubFile(buf, filename, sub)  # 将buf写入子文件中
                    buf = [line]  # 初始化下一个子文件的buf，第一行为*开头的

        # 最后一个文件，文件行数可能不足指定的数
        if len(buf) != 0:
            sub = mk_SubFile(buf, filename, sub)
    print("ok")


if __name__ == '__main__':
    split_By_LineCount(source_file, file_count)  # 要分割的文件名和每个子文件的行数