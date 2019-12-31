from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# Open dataset from file
fileName = './kosarak.dat'
dataset = [(line.rstrip('\n')).split(' ') for line in open(fileName) if not line.isspace()]
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
# print(te_ary)
df = pd.DataFrame(te_ary, columns=te.columns_)
# print(df)
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
print(frequent_itemsets)


# connect to gcp
# gcloud compute ssh --project possible-symbol-263714 --zone us-central1-a instance-1