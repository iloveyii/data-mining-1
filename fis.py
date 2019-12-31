from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import numpy as np
import math
# # dropping null value columns to avoid errors
# data.dropna(inplace = True)
MIN_SUPP = 0.3
DATA_SET_SIZE = 2000


def fis(dataset, split=DATA_SET_SIZE):
    global MIN_SUPP
    global itemsets
    te = TransactionEncoder()
    rows_count = len(dataset)
    max = rows_count
    print('Max : ', max, range(0, max, split))
    dic = {}
    for count in range(0, max, split):
        print('.', end='')
        subset = dataset[count:count + split]
        # print(len(subset))
        te_ary = te.fit(subset).transform(subset)
        # print(te_ary)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        # print(df)
        frequent_itemsets = apriori(df, min_support=MIN_SUPP, use_colnames=True)
        # save this to global variable
        # print(frequent_itemsets.to_json(orient='records'))
        # print('df len', len(frequent_itemsets))
        support = frequent_itemsets.support
        set = frequent_itemsets.itemsets

        for i in range(0, len(frequent_itemsets)):
            separator = '-'
            str_set = separator.join(sorted(list(set.iloc[i])))
            # print(support.iloc[i], str_set)
            if str_set in dic:
                dic[str_set].append(support.iloc[i])
            else:
                dic[str_set] = [support.iloc[i]]

    # print(dic)
    print('')
    for k,v in dic.items():
        print(k, round(np.average(v), 4))

def oneDataset(dataset):
    global MIN_SUPP
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    # print(te_ary)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    # print(df)
    frequent_itemsets = apriori(df, min_support=MIN_SUPP, use_colnames=True)
    print(frequent_itemsets)


# Apple(A) Banana(B) Carrot(C) Dill(D) Ember(E)
dataset = [['A', 'B'],
           ['A', 'B', 'C', 'D'],
           ['A', 'E'],
           ['A', 'B', 'E']]
# Open dataset from file
fileName = './kosarak.dat'
lineList = [(line.rstrip('\n')).split(' ') for line in open(fileName) if not line.isspace()]
# print(lineList[0:15])
dataset = lineList[0:50000]
# print(f'Number of rows : {len(lineList)}')

# print('ONE Dataset')
# oneDataset(dataset)

fis(dataset, DATA_SET_SIZE)
# print('Multi Dataset')

'''
for k,v in itemsets.items():
    print(k)
    print('-------')
    print(v)
'''

# averageAllSets()