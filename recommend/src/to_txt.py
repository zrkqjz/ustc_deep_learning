# SASRec: Self-Attentive Sequential Recommendation原文地址：https://arxiv.org/abs/1808.09781
# 我使用的开源pytorch实现:https://github.com/pmixer/SASRec.pytorch/tree/master
# 这个实现中用0来填充交互序列，但竞赛数据的user和item都从0开始编号，因此我在数据预处理时将item_id和user_id都加1，以便0可以用来填充

import pandas as pd
import os

def main():
    df = pd.read_csv('../data/train_dataset.csv')
    df['user_id'] = df['user_id'] + 1
    df['item_id'] = df['item_id'] + 1
    df.to_csv('../data/goodbooks.txt', sep=',', index=False, header=False)

""" data = pd.read_csv('./data/train_dataset.csv', encoding='utf-8')
# data['user_id']=data['user_id']+1
# data['item_id']=data['item_id']+1
print(data)
data.to_csv('./data/ml-1m.txt', index=0, header=0,encoding='utf-8') """

if __name__ == '__main__':
    main()