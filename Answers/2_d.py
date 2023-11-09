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

# Filter the rules with confidence greater than or equal to 60%
rules_filtered = assoc_rules[assoc_rules['confidence'] >= 0.6]
rules_filtered.loc[:, 'expected_confidence'] = rules_filtered['antecedent support'] * rules_filtered['confidence']
print(rules_filtered [['antecedents', 'consequents', 'support', 'confidence', 'expected_confidence', 'lift']])

rules_filtered.to_csv('2_doutput.csv')



