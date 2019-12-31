from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import math
# # dropping null value columns to avoid errors
# data.dropna(inplace = True)
MIN_SUPP = 0.3
DATA_SET_SIZE = 25000
itemsets = {}


def averageAllSets():
    dic_average = {}
    for k, v in itemsets.items():
        # print(k)
        # print('-----')
        for index, row in v.iterrows():
            items_ary = sorted(list(row['itemsets']))
            separator = '-'
            str_items = separator.join(items_ary)
            print(index, row['support'], str_items)
            if str_items in dic_average:
                da = dic_average[str_items]
                print(da)
                da.append(row['support'])
                dic_average[str_items] = da
            else:
                print(0)
                dic_average[str_items] = []

            continue
            if str_items in dic_average:
                # print(str_items + ' exists in dic')
                dic_average[str_items] = {'support': dic_average[str_items]['support'] + row['support'],
                                          'count': 1 + dic_average[str_items]['count']}
            else:
                dic_average[str_items] = {'support': row['support'], 'count': 1}

    print(dic_average)
    return False
    print('K  V')
    print('----')
    for k, v in dic_average.items():
        if v['count'] != DATA_SET_SIZE:
            print(k, v['support'] / (DATA_SET_SIZE - v['count'] + 1))
        else:
            print(k, v['support'] / v['count'])


def fis(dataset, split=DATA_SET_SIZE):
    global MIN_SUPP
    global itemsets
    te = TransactionEncoder()
    rows_count = len(dataset)
    max = rows_count
    print('Max : ', max, range(0, max, split))
    dic = {}
    for count in range(0, max, split):
        print(count)
        subset = dataset[count:count + split]
        #print(subset)
        te_ary = te.fit(subset).transform(subset)
        # print(te_ary)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        # print(df)
        frequent_itemsets = apriori(df, min_support=MIN_SUPP, use_colnames=True)
        # save this to global variable
        itemsets[count] = frequent_itemsets
        print(frequent_itemsets.to_json(orient='records'))


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
oneDataset(dataset)

fis(dataset, DATA_SET_SIZE)
print('Multi Dataset')

for k,v in itemsets.items():
    print(k)
    print('-------')
    print(v)

averageAllSets()