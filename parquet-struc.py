import pandas as pd
df = pd.read_parquet("data/petitions_with_topics_allk.parquet")
print(df.head(20))
