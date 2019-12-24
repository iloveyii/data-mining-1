from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# Apple(A) Banana(B) Carrot(C) Dill(D) Ember(E)
dataset = [['A', 'B'],
           ['A', 'B', 'C', 'D'],
           ['A', 'E'],
           ['A', 'B', 'E']]
# Open dataset from file
fileName = './kosarak.dat'
lineList = [line.rstrip('\n') for line in open(fileName) if not line.isspace()]
print(lineList[0:5])

te = TransactionEncoder()
te_ary = te.fit(lineList).transform(lineList)
# print(te_ary)  # true false
# print(te_ary.astype("int"))
df = pd.DataFrame(te_ary, columns=te.columns_)
# print(df)
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
print(frequent_itemsets)
