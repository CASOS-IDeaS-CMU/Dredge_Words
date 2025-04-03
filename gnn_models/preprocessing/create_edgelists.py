import pandas as pd
import os
import numpy as np
from ast import literal_eval
from urllib.parse import urlsplit, urlunsplit
from tqdm import tqdm

files = os.listdir('covtwit/processed')
files = ['covtwit/processed/' + file for file in files]
valid_urls = pd.read_csv('covtwit/agg/top_60k_urls.csv')
valid_urls['url_short'] = valid_urls['url_short'].str.lower()
valid_urls = valid_urls['url_short'].unique().tolist()

user_mapper = pd.read_pickle('covtwit/user_maps/user_mapper.pkl')
user_mapper = user_mapper[['orig_idx', 'corrupt_idx']]
#user_mapper.columns = [['orig_user_idx', 'corrupt_idx']]

## Extract user-to-url edgelist
for idx, file in tqdm(enumerate(files)):
    if idx == 0:
        df = pd.read_csv(file)[['user', 'mentions', 'url_short']]
        df['url_short'] = df['url_short'].str.lower()
        df = df[df['url_short'].isin(valid_urls)]
        df = df.merge(user_mapper, how = 'left', left_on = 'user', right_on='orig_idx').dropna(subset = 'corrupt_idx')
        
        url_edgelist = df[['url_short', 'corrupt_idx']]
        del df
        url_edgelist = url_edgelist.groupby(['url_short', 'corrupt_idx']).size().reset_index()
        url_edgelist.columns = ['url_short', 'corrupt_idx', 'count']
    else:
        df = pd.read_csv(file)[['user', 'mentions', 'url_short']]
        df['url_short'] = df['url_short'].str.lower()
        df = df[df['url_short'].isin(valid_urls)]
        df = df.merge(user_mapper, how = 'left', left_on = 'user', right_on='orig_idx').dropna(subset = 'corrupt_idx')
        ueb = df[['url_short', 'corrupt_idx']]
        del df
        ueb = ueb.groupby(['url_short', 'corrupt_idx']).size().reset_index()
        ueb.columns = ['url_short', 'corrupt_idx', 'count']
        
        url_edgelist = pd.concat([url_edgelist, ueb])
        del ueb
        url_edgelist = url_edgelist.groupby(['url_short','corrupt_idx']).agg(
            count = ('count','sum'),
            ).reset_index()
        print("n_rows: ", url_edgelist.shape[0])
        

url_edgelist.to_csv('edgelists/user_to_url.csv', index = False)
observed_users = user_mapper['orig_idx'].tolist()

## extract user-to-(observed)-user edgelist
for idx, file in tqdm(enumerate(files)):
    if idx == 0:
        df = pd.read_csv(file)[['user', 'mentions', 'url_short']]
        df['url_short'] = df['url_short'].str.lower()
        df = df[df['url_short'].isin(valid_urls)]
        df = df[df['mentions'].isin(observed_users)]
        
        df = df.merge(user_mapper, how = 'left', left_on = 'user', right_on='orig_idx').dropna(subset = 'corrupt_idx')
        df = df[['mentions', 'corrupt_idx']]
        df.columns = ['mentions', 'authors']
        df = df.merge(user_mapper, how = 'left', left_on = 'mentions', right_on='orig_idx').dropna(subset = 'corrupt_idx')
        df = df[['authors', 'corrupt_idx']]
        df.columns = ['author', 'mentioned']

        auth_edgelist = df.groupby(['author', 'mentioned']).size().reset_index()
        del df
        auth_edgelist.columns = ['author', 'mentioned', 'count']
    else:
        df = pd.read_csv(file)[['user', 'mentions', 'url_short']]
        df = df[df['url_short'].isin(valid_urls)]
        df = df[df['mentions'].isin(observed_users)]
        
        df = df.merge(user_mapper, how = 'left', left_on = 'user', right_on='orig_idx').dropna(subset = 'corrupt_idx')
        df = df[['mentions', 'corrupt_idx']]
        df.columns = ['mentions', 'authors']
        df = df.merge(user_mapper, how = 'left', left_on = 'mentions', right_on='orig_idx').dropna(subset = 'corrupt_idx')
        df = df[['authors', 'corrupt_idx']]
        df.columns = ['author', 'mentioned']
        
        ueb = df.groupby(['author', 'mentioned']).size().reset_index()
        del df
        ueb.columns = ['author', 'mentioned', 'count']
        
        auth_edgelist = pd.concat([auth_edgelist, ueb])
        del ueb
        auth_edgelist = auth_edgelist.groupby(['author','mentioned']).agg(
            count = ('count','sum'),
            ).reset_index()
        print("n_rows: ", auth_edgelist.shape[0])

# originally udf had 842,154 urls

addl_labs = pd.read_csv('covtwit/addl_labs/addl_labs2.csv')
addl_labs = addl_labs[['url_short', 'user', 'text_idx', 'mentions', 'count']]
user_mapper = user_mapper[['orig_idx', 'corrupt_idx']]
addl_labs.columns = ['url_short', 'orig_idx', 'text_idx', 'mentions', 'count']
addl_labs = addl_labs.merge(user_mapper, how='left', right_on='orig_idx', left_on= 'orig_idx')
addl_labs = addl_labs[['url_short', 'text_idx', 'mentions', 'corrupt_idx', 'count']]
addl_labs.columns = ['url_short', 'text_idx', 'orig_idx', 'author', 'count']
addl_labs = addl_labs.merge(user_mapper, how='left', right_on='orig_idx', left_on= 'orig_idx')
addl_labs = addl_labs[['url_short', 'text_idx', 'author', 'count', 'corrupt_idx']]
addl_labs.to_csv('covtwit/addl_labs/addl_labs3.csv', index = False)

a2m = pd.read_csv('edgelists/user_to_url.csv')
uh = addl_labs[['author', 'url_short']]
auth2urladdl = uh.groupby(['author', 'url_short']).size().reset_index()
auth2urladdl.columns = ['corrupt_idx', 'url_short', 'count']
a2m = pd.concat([a2m, auth2urladdl])
#a2m.to_csv('edgelists/user_to_url.csv', index = False)
del a2m

# repeat with mention edges
a2m = pd.read_csv('edgelists/auth_to_mentioned.csv')
uh = addl_labs[['author', 'corrupt_idx']].dropna()
auth2m = uh.groupby(['author', 'corrupt_idx']).size().reset_index()
auth2m.columns = ['author', 'mentioned', 'count']
a2m = pd.concat([a2m, auth2m])
a2m.to_csv('edgelists/user_to_mentioned.csv', index = False)


#url to url
urlmapper = pd.read_csv('covtwit/agg/mappers/urlmapper.csv')
el1 = pd.read_csv('edgelists/url_el/filtered_backlinks.csv')
el2 = pd.read_csv('edgelists/url_el/seo_sm_addl_links.csv')
el = pd.concat([el1, el2])
# some of the urls in the old (ICSWSM paper) edge list aren't in urlmapper. We can drop those for now

keep =el[el.domain_to.isin(urlmapper['url_short'])]
keep.to_csv('edgelists/url_to_url.csv', index = False)


#df = pd.read_csv('edgelists/user_to_url.csv')
#df.domain_to.nunique()
#df.domain_from.nunique()