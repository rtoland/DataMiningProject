import pandas as pd

df = pd.read_json('review-Colorado.json', lines=True, chunksize=10000)
drops = 0
for chunk in df:
    chunk = chunk.drop(['pics', 'resp'], axis=1)
    l = len(chunk)
    chunk = chunk.dropna()
    d = len(chunk)
    drops += l - d
print(drops)
