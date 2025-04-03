import pandas as pd
import ast
from tqdm import tqdm
import os
from glob import glob
from urllib.parse import urlsplit

import subprocess
import sys

#hm = pd.read_csv('./covtwitter/extracted_url_results/processed/batch1/NewsURLs_virus_2020_1_29.csv')
df = pd.read_csv('networks/urldf1_batch1.csv')
b1_url_counts = df.url.value_counts().reset_index()
b1_url_counts.columns = ['url', 'count']
df = pd.read_csv('networks/urldf1_batch2.csv')
b2_url_counts = df.url.value_counts().reset_index()
b2_url_counts.columns = ['url', 'count']
df = pd.read_csv('networks/urldf1_batch3.csv')
b3_url_counts = df.url.value_counts().reset_index()
b3_url_counts.columns = ['url', 'count']

df = pd.concat([b1_url_counts, b2_url_counts, b3_url_counts])
df['url'] = df['url'].str.lower()
out=df.groupby('url')['count'].sum()
out = out.reset_index()

orig = pd.read_csv('dawn_batch1.csv')
orig['url'] = orig['url'].str.lower()

omerged = orig.merge(out, how= 'left', right_on='url', left_on='url')
omerged = omerged.fillna(0)
omerged['count'] = omerged['count'].astype(int)
omerged.to_csv('domain_counts_covtwit.csv', index = False)


out.sort_values('count')


BATCH = 'batch3'

# if there are .json files, we need to convert them to .jsonl
files = [file for file in os.listdir(BATCH) if file.endswith('.json')]
files = [BATCH + '/' + file for file in files]
for file in files:
    subprocess.run(['twarc2', 'flatten', f'{file}', f'{file[:-5]}.jsonl'])
    subprocess.run(['rm', f'{file}'])


def clean_dict_fields(df):
    """Get rid of some of the dictionary columns and extract fields of interest

    Args:
        df (_type_): _description_
    """
    
    quoted_user, tag_list = [], []
    user_mentioned_list = []
    url_list, idx_list = [], []
    user, description = [], []
    rt_text = []
    if df.is_quote_status.sum() > 0:
        for idx, i in enumerate(df.quoted_status.tolist()):
            if type(i) is dict:
                quoted_user.append(str(i['user']['id']))
            else:
                quoted_user.append(None)
    else:
        for i in enumerate(df.entities.tolist()):
            quoted_user.append(None)
    
    for idx, i in enumerate(df.retweeted_status.tolist()):
        # retweeted text is trunctated outside of the retweet object, so we need to extract it
        if type(i) is dict:
            extended = i.get('extended_tweet')
            if extended is not None:
                rt_text.append(extended.get('full_text'))
            else:
                rt_text.append(str(i.get('text')))
        else:
            rt_text.append(df.iloc[idx]['text'])
    else:
        pass
    
    for idx, i in enumerate(df.user.tolist()):
        user.append(str(i['id']))
        description.append(str(i['description']))

    for idx, entity in enumerate(df.entities.tolist()):
        try:
            entity = ast.literal_eval(str(entity))
            
            # hashtags
            hashtags = entity.get('hashtags')
            if len(hashtags) > 0:
                tags = [tag.get('tag') for tag in hashtags]
            else:
                tags = None
            
            # mentioned users
            user_mentions = entity.get('user_mentions')
            if len(user_mentions) > 0:
                users = [user.get('id') for user in user_mentions]
            else:
                users = None
            
            # urls
            all_urls = entity.get('urls')
            if len(all_urls) > 0:
                urls = [url.get('expanded_url') for url in all_urls]
            else:
                urls = None
            
            tag_list.append(tags)
            user_mentioned_list.append(users)
            url_list.append(urls)
            idx_list.append(idx)
        except:
            print(idx)
    
    mentioned_and_quoted = []
    for i, j in zip(user_mentioned_list, quoted_user):
        if i is None:
            mentioned_and_quoted.append(j)
        elif j is None:
            mentioned_and_quoted.append(i)
        else:
            if type(i) is str:
                i = [i]
            if type(j) is str:
                j = [j]
            mentioned_and_quoted.append(i + j)
    
    tweet_attrs = pd.DataFrame({'user':user,
                                'text':rt_text,
                                'description':description,
                                #'quoted':quoted_user,
                                'hashtags':tag_list,
                                'user_mentions':mentioned_and_quoted,
                                'urls':url_list})
    
    return tweet_attrs



# read in json files
files = [BATCH + '/' + file for file in os.listdir(BATCH) if file.endswith('.jsonl')]
# save processed data to ./processed/* as csv's
for file in files:
    test = pd.read_json(file, lines =True, chunksize = 1000, dtype='object')
    dfs = []
    shapes = []
    for idx, i in enumerate(tqdm(test)):
        tweets = clean_dict_fields(i)
        tweets['tweet_id'] = i['id'].tolist()
        dfs.append(tweets)
        shapes.append(i.shape[0]) 
        
    df = pd.concat(dfs)
    tid_check = df.isna().sum()
    if tid_check['tweet_id'] > 0:
        print(f'There are missing tweet ids in {file}')
        break
    df.to_csv(f'processed/{file[:-5]}csv', index = False)

# clean up url networks
files = [file for file in os.listdir('./processed/' + BATCH) if file.endswith('.csv')]
files = ["./processed/" + BATCH + '/' + file for file in files]
urldfs = []

for file in tqdm(files):
    try:
        df = pd.read_csv(file, dtype=object, lineterminator='\n').dropna(subset = 'urls')
    except:
        df = pd.read_csv(file, dtype=object, engine = 'python').dropna(subset = 'urls')

    url_list = []
    user_id_list = []
    text_list = []
    for idx, i in enumerate(df.urls.tolist()):
        try:
            urls = ast.literal_eval(i)
            base_urls = [urlsplit(url).netloc for url in urls]
            for j in base_urls:
                url_list.append(j)
                user_id_list.append(df.user.iloc[idx])
                text_list.append(df.text.iloc[idx])
        except:
            pass

    urlnet_chunk = pd.DataFrame({'url':url_list, 'user_id':user_id_list, 'text':text_list})
    urldfs.append(urlnet_chunk)
df = pd.concat(urldfs)
df.to_csv('networks/urldf1_' + BATCH + '.csv', index = False)
del urldfs
del df

## Repeat for user networks
files = [file for file in os.listdir('./processed/' + BATCH) if file.endswith('.csv')]
files = ["./processed/"+ BATCH + '/' + file for file in files]
userdfs = []

for file in tqdm(files):
    try:
        # drop tweets where users were neither quoted nor mentioned
        df = pd.read_csv(file, dtype=object, lineterminator='\n').dropna(subset = ['user_mentions'])
    except:
        # drop tweets where users were
        df = pd.read_csv(file, dtype=object, engine = 'python').dropna(subset = 'user_mentions')
    mentioned_list = []
    user_id_list = []
    
    for idx, i in enumerate(df.user_mentions.tolist()):
        user_mentions = ast.literal_eval(i)
        if type(user_mentions) is int:
            user_mentions = [str(user_mentions)]
        elif type(user_mentions) is str:
            user_mentions = [user_mentions]
        else:
            continue
        user_mentions = [str(user) for user in user_mentions]
        for j in user_mentions:
            mentioned_list.append(j)
            user_id_list.append(df.user.iloc[idx])

    final_urls = pd.DataFrame({'mentioned':mentioned_list, 'user_id':user_id_list})
    userdfs.append(final_urls)

df = pd.concat(userdfs)
df.columns = ['target', 'source']
df.to_csv('./networks/mention_network_' + BATCH + '.csv', index = False)
del df


## get url_network + mention_network
# clean up url networks
files = [file for file in os.listdir('./processed/' + BATCH) if file.endswith('.csv')]
files = ["./processed/" + BATCH + '/' + file for file in files]
urldfs = []
for file in tqdm(files):
    try:
        df = pd.read_csv(file, dtype=object, lineterminator='\n').dropna(subset = 'urls')
    except:
        df = pd.read_csv(file, dtype=object, engine = 'python').dropna(subset = 'urls')

    url_list = []
    user_id_list = []
    mentioned_list = []
    for idx, i in enumerate(df.urls.tolist()):
        try:
            urls = ast.literal_eval(i)
            base_urls = [urlsplit(url).netloc for url in urls]
            for j in base_urls:
                url_list.append(j)
                user_id_list.append(df.user.iloc[idx])
                mentioned_list.append(df.user_mentions.iloc[idx])
        except:
            pass

    urlnet_chunk = pd.DataFrame({'url':url_list, 'user_id':user_id_list, 'mentioned':mentioned_list})
    urldfs.append(urlnet_chunk)
df = pd.concat(urldfs)
df.to_csv('networks/url_network_' + BATCH + '.csv', index = False)

# create final networks
df = pd.read_csv('networks/url_network.csv')
df = df[df.url != "twitter.com"]
mentioned_df = df.copy().dropna()
mentioned_df.mentioned = mentioned_df.mentioned.apply(lambda x: ast.literal_eval(x))

gn = pd.merge(hm, labels, left_on = 'url', right_on='url', how = 'left')
# subset by labeled domains
el = pd.read_csv('./networks/url_mentions.csv', dtype=object)
labels = pd.read_csv('../../twitter/final/final_labels.csv', dtype=object)
l2 = pd.read_csv('networks/domain_list_clean.csv')
labels.url = labels.url.str.lower()
urls_dirty = l2.url.str.lower().tolist()
rest_urls = labels.url.str.lower().unique().tolist()
filtered_urls = list(set(urls_dirty + rest_urls))
el.url = el.url.str.lower()

#filtered_urls = labels.url.unique().tolist()
df = el[el.url.isin(filtered_urls)]
mentioned_df = df.copy().dropna()
mentioned_df.mentioned = mentioned_df.mentioned.apply(lambda x: ast.literal_eval(x))

source = []
target = []
for idx, x in enumerate(mentioned_df.mentioned.tolist()):
    if type(x) is int:
        x = [str(x)]
    source_node = mentioned_df.user_id.iloc[idx]
    for user in x:
        source.append(source_node)
        target.append(user)
    assert len(source) == len(target), f"source and target length mismatch at {idx}"
mentioned_user_df = pd.DataFrame({'source':source, 'target':target})
mentioned_user_df.to_csv('./v2/filtered_mention_network_final.csv', index = False)

source = []
target = []
for idx, x in enumerate(mentioned_df.mentioned.tolist()):
    if type(x) is int:
        x = [str(x)]
    target_node = mentioned_df.url.iloc[idx]
    for user in x:
        source.append(user)
        target.append(target_node)
    assert len(source) == len(target), f"source and target length mismatch at {idx}"
mentioned_user_df = pd.DataFrame({'source':source, 'target':target})
mentioned_user_df.to_csv('./v2/filtered_mentioned_to_url.csv', index = False)

filtered_user_url = df[['user_id', 'url']]
filtered_user_url.columns = ['source', 'target']
filtered_user_url.to_csv('v2/filtered_user_to_url.csv', index = False)

mtu = pd.read_csv('v2/filtered_mentioned_to_url.csv')
fmn = pd.read_csv('v2/filtered_mention_network_final.csv')
fuu = pd.read_csv('v2/filtered_user_to_url.csv')

mtu['type'] = 'target2url'
fmn['type'] = 'auth2target'
fuu['type'] = 'auth2url'

final = pd.concat([mtu, fmn, fuu])
final.to_csv('v2/final_network_userlinks.csv', index = False)

urls = pd.read_csv('ahrefs/urls.csv')
urls_backlinks = pd.read_csv('ahrefs/urls_backlinks.csv')

# give 23 cols
unique_urls = fuu.target.unique().tolist()
label_sanity = labels[labels.url.isin(unique_urls)]
sml = label_sanity.label.value_counts().reset_index().sort_values('index')
sml.columns = ['label', 'unique_urls']



# high- level network stats


df = pd.read_csv('v2/final_network_userlinks.csv')

## Repeat for user networks
files = [file for file in os.listdir('./processed') if file.endswith('.csv')]
files = ["./processed/" + file for file in files]

users = []
n_posts = []

for file in tqdm(files):
    try:
        # drop tweets where users were neither quoted nor mentioned
        df = pd.read_csv(file, dtype=object, lineterminator='\n')#.dropna(subset = ['user_mentions'])
    except:
        # drop tweets where users were
        df = pd.read_csv(file, dtype=object, engine = 'python')#.dropna(subset = 'user_mentions')

    n_posts.append(df.shape[0])
    users.append(df.user.unique())

flat_list = [item for sublist in users for item in sublist]
len(set(flat_list)) # 3993538