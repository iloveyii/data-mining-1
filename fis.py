from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import math

average = []

# # dropping null value columns to avoid errors
# data.dropna(inplace = True)
dic = {}
MIN_SUPP = 0.000001
itemsets = {}


def addaverage2(frequent_itemsets):
    global dic
    for index, row in frequent_itemsets.iterrows():
        itemsets = row['itemsets']
        support = row['support']
        print(support, itemsets)
        continue
        if itemsets in dic:
            print('exist')
            # take average
            oldSupport = dic[itemsets]
            print('oldSupport', oldSupport)
            avgSupport = (oldSupport + support) / 2
            dic[itemsets] = avgSupport
        else:
            dic[itemsets] = support
        print(dic)


def fis(dataset, split=1000):
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
        print(subset)
        te_ary = te.fit(subset).transform(subset)
        # print(te_ary)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        # print(df)
        frequent_itemsets = apriori(df, min_support=MIN_SUPP, use_colnames=True)
        # save this to global variable
        itemsets[count] = frequent_itemsets

        listSupport = list(frequent_itemsets['support'])
        listItemsets = list(frequent_itemsets['itemsets'])
        # print(listSupport, listItemsets)
        # addaverage(frequent_itemsets)
        max = len(listSupport)
        for i in range(0, max):
            # print(listSupport[i], listItemsets[i])
            if listItemsets[i] in dic:
                dicListItemsets = dic[listItemsets[i]]
                dic[listItemsets[i]] = {'support': dicListItemsets['support'] + listSupport[i],
                                        'count': dicListItemsets['count'] + 1}
            else:
                dic[listItemsets[i]] = {'support': listSupport[i], 'count': 1}

    # print(dic)
    # for key, value in dic.items():
    # print(f'{key} {value} - ', value['support']/value['count'])
    return False

    sets = 0
    dic = {}
    for count in range(0, max, split):
        # print(count)
        sets += 1

        # print(sets)
        te_ary = te.fit(dataset[count:count + split]).transform(dataset[count:count + split])
        # print(te_ary)  # true false
        # print(te_ary.astype("int"))
        df = pd.DataFrame(te_ary, columns=te.columns_)
        # print(df)
        frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
        # addaverage(frequent_itemsets)
        print(frequent_itemsets)

        for index, row in frequent_itemsets.iterrows():
            itemsets = row['itemsets']
            support = row['support']
            print(support, itemsets)
            if itemsets in dic:
                print('exist')
                # take average
                oldSupport = dic[itemsets]
                avgSupport = (oldSupport + support) / 2
                dic[itemsets] = avgSupport
            else:
                dic[itemsets] = {'support': support, count: 1}
            print(dic)


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
# lineList = [(line.rstrip('\n')).split(' ') for line in open(fileName) if not line.isspace()]
# print(lineList[0:15])
# dataset = lineList[0:2000]
# print(f'Number of rows : {len(lineList)}')
print('ONE Dataset')
oneDataset(dataset)

fis(dataset, 2)
print('Multi Dataset')

dic_average = {}
for k, v in itemsets.items():
    print(k)
    print('-----')
    for index, row in v.iterrows():
        items_ary = sorted(list(row['itemsets']))
        separator = '-'
        str_items = separator.join(items_ary)
        print(index, row['support'], str_items)
        if str_items in dic_average:
            # print(str_items + ' exists in dic')
            dic_average[str_items] = {'support': dic_average[str_items]['support'] + row['support'], 'count': 1 + dic_average[str_items]['count']}
        else:
            dic_average[str_items] = {'support': row['support'], 'count': 1}

print('K  V')
print('----')
for k, v in dic_average.items():
    if v['count'] > 1:
        print(k, v['support']/v['count'])
