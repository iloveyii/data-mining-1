from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import numpy as np

MIN_SUPP = 0.2  # Support
DATA_SET_SIZE = 5000  # SEGMENT size of data


def printSet(dic):
    print('')
    for k, v in dic.items():
        print(k, round(np.average(v), 4))


def addToDict(frequent_itemsets):
    support = frequent_itemsets.support
    set = frequent_itemsets.itemsets

    for i in range(0, len(frequent_itemsets)):
        separator = '-'
        str_set = separator.join(sorted(list(set.iloc[i])))
        if str_set in dic:
            dic[str_set].append(support.iloc[i])
        else:
            dic[str_set] = [support.iloc[i]]


def fis(dataset, split=DATA_SET_SIZE):
    global MIN_SUPP
    te = TransactionEncoder()
    rows_count = len(dataset)
    maximum = rows_count
    print('maximum : ', maximum, range(0, maximum, split))
    dic = {}
    for count in range(0, maximum, split):
        print('.', end='')
        subset = dataset[count:count + split]
        te_ary = te.fit(subset).transform(subset)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        frequent_itemsets = apriori(df, min_support=MIN_SUPP, use_colnames=True)
        support = frequent_itemsets.support
        set = frequent_itemsets.itemsets

        for i in range(0, len(frequent_itemsets)):
            separator = '-'
            str_set = separator.join(sorted(list(set.iloc[i])))
            if str_set in dic:
                dic[str_set].append(support.iloc[i])
            else:
                dic[str_set] = [support.iloc[i]]

    # print(dic)
    printSet(dic)


'' #### RUN CODE ##### ''
# Open dataset from file
fileName = './kosarak.dat'
lineList = [(line.rstrip('\n')).split(' ') for line in open(fileName) if not line.isspace()]

fis(lineList, DATA_SET_SIZE)
