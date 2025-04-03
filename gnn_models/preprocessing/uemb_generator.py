import pandas as pd
import numpy as np
from tqdm import tqdm
import os

user_df_orig = pd.read_pickle('covtwit/agg/mappers/user_to_text.pkl')
lm2 = pd.read_csv('covtwit/agg/mappers/langmapperv2.csv')
textmapper = lm2.set_index('text_idx').to_dict()['new_idx']
user_df_orig['seq_idx'] = list(range(user_df_orig.shape[0]))
unique_text_vals = list({textmapper[x] for l in user_df_orig['text_idx'].tolist() for x in l})


#del user_df_orig

files = os.listdir('covtwit/embeddings/text/')
files = ['./covtwit/embeddings/text/' + file for file in files]
lencache = []
for i in range(len(files)):
    b0 = np.load(f'covtwit/embeddings/text/batch_{i}.npy')
    lencache.append(b0.shape[0])


batch_idx_mapper = {}
running_total = 0
for i in range(len(lencache)):
    if i == 0:
        batch_idx_mapper[i] = running_total
    else:
        running_total += lencache[i-1]
        batch_idx_mapper[i] = running_total

bm_total = 0
batch_max_idx = []
for i in range(len(lencache)):
    bm_total += lencache[i]
    batch_max_idx.append(bm_total)
    
# less than logic requires this or it that tweet will throw an error
batch_max_idx[-1] += 1
batch_max_idx = np.array(batch_max_idx)

#user_df_orig = user_df_orig.sort_values('corrupt_idx')
#user_df_orig = user_df_orig.reset_index().reset_index()
#user_df_orig = user_df_orig[['level_0', 'orig_idx', 'corrupt_idx', 'text_idx']]
#user_df_orig.columns = ['uidx', 'orig_idx', 'corrupt_idx', 'text_idx']

### better approach
user_df_orig = user_df_orig[['seq_idx', 'text_idx']]
list_df = np.array_split(user_df_orig, 50)
user_embedding_cache = []
user_ids_cache = []

for chunk in tqdm(list_df): #tqdm(range(list_df)):
    chunk = chunk.explode('text_idx')[['seq_idx', 'text_idx']]
    chunk.text_idx = chunk.text_idx.map(textmapper)
    # maybe forbid last tweet- causing index issues..
    chunk = chunk.groupby('seq_idx').sample(10)

    user_id = chunk['seq_idx'].tolist()

    text_to_batch_mapper = {}
    cvals = chunk['text_idx'].values
    
    for text in cvals:
        text_to_batch_mapper[text] = np.where(text < batch_max_idx)[0][0]
    batch_helper = pd.DataFrame(text_to_batch_mapper, index = [0]).T.reset_index()
    batch_helper.columns = ['text_idx', 'batch_n']
    batch_helper['bnorm'] = batch_helper['batch_n'].map(batch_idx_mapper)
    batch_helper['index_norm'] = batch_helper['text_idx'] - batch_helper['bnorm']
    batch_helper = chunk.merge(batch_helper, how = 'left', left_on='text_idx', right_on = 'text_idx')
    batch_helper = batch_helper.sort_values('batch_n')
    print(batch_helper['index_norm'].min())
    np_cache = None
    for bnum in batch_helper['batch_n'].unique().tolist():
        b0 = np.load(f'covtwit/embeddings/text/batch_{bnum}.npy')
        if bnum == 14:
            b21 = batch_helper[batch_helper['batch_n'] == 14].copy()
            b21['index_norm'] = b21['index_norm'] - 1
            batch_idx = b21['index_norm'].values
        else:
            batch_idx = batch_helper[batch_helper['batch_n'] == bnum]['index_norm'].values
        b0 = b0[batch_idx]
        if np_cache is None:
            np_cache = b0
        else:
            np_cache = np.vstack([np_cache, b0])
    np_cache = pd.DataFrame(np_cache)
    np_cache['seq_idx'] = batch_helper['seq_idx'].tolist()
    user_vals = np_cache.groupby('seq_idx').mean()
    users = user_vals.reset_index()['seq_idx'].tolist()
    user_vals = user_vals.to_numpy()
    
    user_embedding_cache.append(user_vals)
    user_ids_cache.append(users)
    
    del user_vals 
    del users

out_1 = np.vstack(user_embedding_cache)
np.save(f'covtwit/embeddings/textv2/embeddings_550k', out_1)

users_final = np.concatenate(user_ids_cache)
users_final = pd.DataFrame({'uid':users_final})
users_final.to_csv('covtwit/embeddings/textv2/user_850k.csv', index=False)

## re-index user embeddings
user_re_indexer = pd.read_pickle('covtwit/user_maps/processed_batch_0.pkl')
user_re_indexer = user_re_indexer.reset_index().reset_index()
user_re_indexer.columns = ['new_idx', 'orig_idx', 'corrupt_idx', 'text_idx', 'lens']
user_re_indexer.to_pickle('covtwit/user_maps/processed_batch_0.pkl')


## BAD WAY
user_embedding_mapper = {}
for i in tqdm(range(user_df_orig.shape[0])):
    texts = sorted(np.random.choice(user_df_orig.iloc[i].text_idx, size=10, replace=False))
    user_id = user_df_orig.iloc[i].uidx
    text_to_batch_mapper = {}
    for text in texts:
        text_to_batch_mapper[text] = np.where(text < batch_max_idx)[0][0]
    batch_helper = pd.DataFrame(text_to_batch_mapper, index = [0]).T.reset_index()
    batch_helper.columns = ['tidx', 'batch_n']
    batch_helper['bnorm'] = batch_helper['batch_n'].map(batch_idx_mapper)
    batch_helper['index_norm'] = batch_helper['tidx'] - batch_helper['bnorm']
    
    running_mean = None
    for bnum in batch_helper['batch_n'].unique().tolist():
        b0 = np.load(f'covtwit/embeddings/text/batch_{bnum}.npy')
        batch_idx = batch_helper[batch_helper['batch_n'] == bnum]['index_norm'].values
        b0 = b0[batch_idx]
        if len(b0.shape) > 1:
            b0 = b0.mean(axis = 0).reshape(1, 512)
        else:
            b0 = b0.reshape(1,512)
        if running_mean is None:
            running_mean = b0
        else:
            running_mean = np.mean(np.vstack([running_mean, b0]), axis = 0).reshape(1,512)
    user_embedding_mapper[user_id] = running_mean
