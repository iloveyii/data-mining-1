from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

average = []


def addaverage(series):
    print(series)
    supportList = series['support'].tolist()
    itemsetsList = series['itemsets'].tolist()
    print(supportList, itemsetsList)
    avg = []
    val=[]
    for item in itemsetsList:
        print(str(item))
        if not item in avg:
            avg.append(item)


    print(avg)
    # check if set exists
    # add average


def fis(dataset, split=1000):
    te = TransactionEncoder()
    rowsCount = len(dataset)
    sets = 0
    for count in range(0, rowsCount + split, split):
        # print(count)
        sets += 1

        # print(sets)
        te_ary = te.fit(dataset[count:count + split]).transform(dataset[count:count + split])
        # print(te_ary)  # true false
        # print(te_ary.astype("int"))
        df = pd.DataFrame(te_ary, columns=te.columns_)
        # print(df)
        frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
        addaverage(frequent_itemsets)
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

dataset = lineList[0:1000]
print(f'Number of rows : {len(lineList)}')
print('Dataset')
fis(dataset)
