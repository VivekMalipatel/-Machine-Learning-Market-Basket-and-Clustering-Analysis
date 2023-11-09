import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file into a pandas dataframe
df = pd.read_csv('/Users/vivekmalipatel/Downloads/Assignmnts/IntrotoMLAssignment/Question/TwoFeatures.csv')

# Plot the x2 versus x1
plt.scatter(df['x1'], df['x2'])
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.show()