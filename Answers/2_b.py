import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

# Load the data into a Pandas DataFrame
data = pd.read_csv("/Users/vivekmalipatel/Downloads/Assignmnts/IntrotoMLAssignment/Question/Groceries.csv")

te = TransactionEncoder()
te_ary = te.fit(data.groupby('Customer')['Item'].apply(list)).transform(data.groupby('Customer')['Item'].apply(list))

# Convert the transactions list into a pandas DataFrame
transactions = pd.DataFrame(te_ary, columns=te.columns_)
nCustomer, nProduct = transactions.shape

frequent_itemsets = apriori(transactions, min_support= 75 / nCustomer, use_colnames=True)
# Print the number of itemsets found
print("Number of itemsets found:", len(frequent_itemsets))

# Find the largest number of items (k) among the itemsets
print("Largest number of items among the itemsets:", frequent_itemsets['itemsets'].apply(lambda x: len(x)).max())

