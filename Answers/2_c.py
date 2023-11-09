import matplotlib.pyplot as plt
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
nCustomer, nProduct = ItemIndicator.shape

# Find the frequent itemsets
lowest_support = 75.0 / nCustomer
frequent_itemsets = apriori(ItemIndicator, min_support = lowest_support, max_len = 4, use_colnames = True)

# Discover the association rules
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)

print("Number of Association Rules : " + str(len(assoc_rules)))

plt.figure(figsize=(10,6), dpi = 200)
plt.scatter(assoc_rules['confidence'], assoc_rules['support'],
            c = assoc_rules['lift'], s = 5**assoc_rules['lift'])
plt.grid(True, axis = 'both')
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.colorbar().set_label('lift')
plt.show()
