import pandas as pd
from functools import reduce

def stream(filename = 'review-Colorado.json', c_size = 1000):
    return pd.read_json(filename, lines=True, chunksize=c_size)

df = stream()

# n = reduce(lambda acc, chunk: acc + chunk.dropna()[chunk.dropna()['text'].astype(str).str.split().str.len() < 5].shape[0] if not chunk.empty else acc, df, 0)

# print(n)

acc = 0

for chunk in df:
    chunk = chunk.drop(['pics', 'resp'], axis=1)
    chunk = chunk.dropna()
    if not chunk.empty:
        acc += chunk[chunk['text'].str.split().str.len() < 5].shape[0]

print(acc)