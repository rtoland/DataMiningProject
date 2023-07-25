import pandas as pd

sus = []

def count(g):
    v_counts = g['name'].value_counts()
    l = v_counts[v_counts > 5].index.tolist()
    if len(l) > 0:
        sus.append((g.time.dt.date.iloc[0], l))

# filename = 'review-Alaska_reduced_v2.json.gz'
filename = 'Alaska.zip'

df = pd.read_csv(filename, compression='zip')

df['time'] = pd.to_datetime(df['time'])

assert(df.time.dtype == 'datetime64[ns]')

groups = df.groupby(pd.Grouper(key='time', freq='D'))

groups.apply(lambda x: count(x))

print(sus)
