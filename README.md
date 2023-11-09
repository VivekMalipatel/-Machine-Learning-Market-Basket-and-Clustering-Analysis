# -Machine-Learning-Market-Basket-and-Clustering-Analysis
CS 484: Introduction to Machine Learning Assignment

# CS 484: Introduction to Machine Learning
## Spring Semester 2023 - Assignment 2

This repository contains the materials for Assignment 2 of the Introduction to Machine Learning course. Below are the questions and instructions for the assignment.

### Question 1: Analyzing the Association Rule (20 points)

You are asked to calculate the Lift of the association rule {Cheese, Wing} ==> {Soda} based on items brought by six friends for a basketball game. The items brought by each friend are listed below:

| Friend | Items                                      |
|--------|--------------------------------------------|
| Andrew | Cheese, Cracker, Soda, Wing                |
| Betty  | Cheese, Soda, Tortilla, Wing               |
| Carl   | Cheese, Ice Cream, Wing                    |
| Danny  | Cheese, Ice Cream, Salsa, Soda, Tortilla   |
| Emily  | Salsa, Soda, Tortilla, Wing                |
| Frank  | Cheese, Cracker, Ice Cream, Wing           |

Calculate the Lift for the given association rule.

### Question 2: Discovering Association Rules (40 points)

Using the `Groceries.csv` file, perform the following tasks:

#### Part (a) - Itemset Analysis (10 points)

- Determine the number of items in the Universal Set.
- Calculate the maximum number of itemsets and association rules that can be found in theory.

#### Part (b) - Market Basket Analysis (10 points)

- Identify itemsets that appear in at least seventy-five (75) customer baskets.
- Report the number of such itemsets and the largest number of items, `k`, among them.

#### Part (c) - Association Rule Generation (10 points)

- Generate association rules with a Confidence metric of at least 1%.
- Plot Support vs. Confidence, using Lift to indicate the marker size and adding a color gradient for Lift.

#### Part (d) - High Confidence Rule Listing (10 points)

- List rules with a Confidence metric of at least 60%.
- Present these rules in a table with Antecedent, Consequent, Support, Confidence, Expected Confidence, and Lift.

### Question 3: Clustering with Rescaled Features (40 points)

Using the `TwoFeatures.csv` file, complete the following clustering tasks:

#### Part (a) - Data Visualization (10 points)

- Plot `x2` against `x1` with gridlines.
- Estimate the number of clusters visually.

#### Part (b) - Optimal Cluster Discovery (10 points)

- Find the optimal number of clusters using the K-Means algorithm with Manhattan distance.
- List the number of clusters, TWCSS, and Elbow values. Plot Elbow Values vs. number of clusters.
- Report the centroids of the optimal clusters.

#### Part (c) - Clustering with Rescaled Data (10 points)

- Rescale `x1` and `x2` and find the optimal number of clusters again.
- List the number of clusters, TWCSS, and Elbow values. Plot Elbow Values vs. number of clusters.
- Provide the centroids of the optimal clusters in the original scale.

#### Part (d) - Analysis of Results (10 points)

- Discuss the differences between the two optimal cluster solutions found.
