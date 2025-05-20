import pandas as pd

df = pd.read_csv('answers_by_article.csv')
count = df['ArticleID'].nunique()

print("Total number of articles processed:", count)