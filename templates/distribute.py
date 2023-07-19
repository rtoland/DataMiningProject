import pandas as pd

df = pd.read_json('review-Colorado', lines=True, chunksize=10000)

n_files = 18
items_per_bucket = [540877, 540876, 540876, 540876, 540876, 540876, 540876, 540876, 540876, 540876, 540876, 540876, 540876, 540876, 540876, 540876, 540876, 540876]
filenames = ['reviews_part' + str(x) + '.json' for x in range(n_files)]
fds = [open(filename, 'a') for filename in filenames]
file_counter = 0
item_counter = 0

for chunk in df:
    chunk = chunk.drop(['pics', 'resp', 'gmap_id'], axis=1)
    chunk = chunk.dropna()
    n_rows = len(chunk)
    for i in range(n_rows):
        if items_per_bucket[file_counter] <= 0:
            file_counter += 1
        f = fds[file_counter]
        json.dump(chunk.loc[i].to_dict(), f)
        items_per_bucket[file_counter] -= 1
