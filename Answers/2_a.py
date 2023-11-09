import pandas

from mlxtend.frequent_patterns import (apriori, association_rules)
from mlxtend.preprocessing import TransactionEncoder

data = pandas.read_csv('/Users/vivekmalipatel/Downloads/Assignmnts/IntrotoMLAssignment/Question/Groceries.csv')


# Convert the Sale Receipt data to the Item List format
ListItem = data.groupby(['Customer'])['Item'].apply(list).values.tolist()

# Convert the Item List format to the Item Indicator format
te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
ItemIndicator = pandas.DataFrame(te_ary, columns=te.columns_)
nCustomer, num_items = ItemIndicator.shape

# Calculate the maximum number of itemsets
max_itemsets = (2**num_items) - 1

# Calculate the maximum number of association rules
max_rules = (3**num_items) - (2**(num_items+1)) + 1

print("Number of items in the Universal Set:", num_items)
print("Maximum number of itemsets:", max_itemsets)
print("Maximum number of association rules:", max_rules)