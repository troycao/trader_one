# coding:utf-8
'''
数据处理类
'''

import os

def filter_data_byLine(path):
    for home, dirs, files in os.walk(path):
        print("#######dir list#######")
        for dir in dirs:
            print(dir)
        print("#######dir list#######")

        print("#######file list#######")
        for filename in files:
            fullname = os.path.join(home, filename)
            file = open(fullname, "r")
            lines = file.readlines()
            temp_lines = lines[:-1]
            file = open(fullname, "w")
            file.writelines(temp_lines)
        print("#######file list#######")


def filter_data_byKey(path, key_word):
    for home, dirs, files in os.walk(path):
        print("#######dir list#######")
        for dir in dirs:
            print(dir)
        print("#######dir list#######")

        print("#######file list#######")
        for filename in files:
            fullname = os.path.join(home, filename)
            file = open(fullname, "rb+")
            lines = file.readlines()
            for str in lines:
                print(str)

        print("#######file list#######")


if __name__ == "__main__":
	#.为遍历当前目录  其他的给绝对路径
    # filter_data_byKey('G:/data/stock/20210418/', '数据来源:通达信')
    filter_data_byLine('G:/data/stock/20210418/')


