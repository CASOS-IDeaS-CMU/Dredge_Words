from lib2to3.pgen2.pgen import DFAState
import os.path as osp
import matplotlib.pyplot as plt
import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.utils.convert import from_networkx
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from gnns.seo_import import *
from sklearn import preprocessing
from torch_geometric.datasets import FakeHeteroDataset
import torch_geometric.transforms as T

# use filtered el/ ol for sm
NETWORK_TYPE = 'sm' #['links', 'sm'] 'sm' = social media users + links
edge_path = 'ahrefs/labeled/backlinks.csv'
sm_edge_path = 'edgelists/user_to_url.csv'
attr_path = 'ahrefs/labeled/attributes.csv'
embedding_path = 'embeddings/emb_test.npy'
seo_only_embedding_path = 'embeddings/emb_seo_only.npy'
user_text_embedding_path = 'covtwit/embeddings/textv2/embeddings_850k.npy'
label_path = 'domain_ratings.csv'
dredge_url_text_path = 'embeddings/dredge/dredge_mappers.csv'
dredge_url_path = 'embeddings/dredge/dredge_twitter_text_feats.npy'
dredge_url_embedding_path = 'embeddings/dredge/dredge_twitter_text_feats.npy'
embedding_path = 'embeddings/dredge/dredge_twitter_text_feats.npy'

bad_drop = True

def import_seo_and_users(edge_path, attr_path, sm_edge_path, label_path):
    """
    import all data- all entites without labels are assigned -2 in this function
    """
    el_seo = pd.read_csv(edge_path).dropna()[['domain_from', 'domain_to']]
    labeldf = pd.read_csv(label_path)
    # discretize labels with quantiles
    labeldf['label'] = pd.qcut(labeldf['pc1'], q=5, labels=[0, 1, 2, 3, 4])
    labeldf = labeldf[['domain', 'label']]
    labeldf.columns = ['url', 'label']
    
    labeldf = labeldf[labeldf.url.isin(el_seo.domain_to)]
    #### charitynavigator.org 
    
    urls_in_twitter = labeldf.url.unique().tolist()
    el_seo = el_seo[el_seo.domain_to.isin(urls_in_twitter)]

    #labeldf['label']= labeldf.label.astype(int).replace({6:1,5:1,4:0,3:0,2:0,1:0,-1:-1})
    # social media data
    el_sm = pd.read_csv(sm_edge_path).drop_duplicates().drop(columns = 'count')
    #el_sm.columns = ['domain_from', 'domain_to']
    #sites w/o labels
    other_sites = el_seo[~el_seo['domain_from'].isin(labeldf.url.unique().tolist())].domain_from.unique()
    
    dredge_domains = pd.read_csv('embeddings/dredge/dredge2domain.csv').dropna()
    dredge_domains = dredge_domains[~dredge_domains['domain'].isin(labeldf.url.unique().tolist())]
    dredge_domains = dredge_domains[~dredge_domains['domain'].isin(other_sites)]
    dredge_urls = dredge_domains.domain.unique().tolist()
    other_sites = other_sites.tolist() + dredge_urls
    
    # with dredge, 573,734 users
    other_sites = pd.DataFrame({'url':other_sites, 'label':-1})
    labeldf = pd.concat([labeldf, other_sites])

    el_sm.columns = ['domain_to', 'domain_from']
    
    # if we wanted only labeled domains: 
    el_sm = el_sm[el_sm['domain_to'].isin(list(set(labeldf.url.tolist() + el_seo.domain_from.tolist())))]

    # drop users who've only linked to a single domain.
    drop_pendulum_urls = el_sm.groupby('domain_to').size().reset_index()
    drop_pendulum_urls.columns = ['user', 'n']
    drop_pendulums_urls = drop_pendulum_urls[drop_pendulum_urls['n']>1]['user'].tolist()
    el_sm = el_sm[el_sm['domain_to'].isin(drop_pendulums_urls)]
    del drop_pendulum_urls
    drop_pendulum_users = el_sm.groupby('domain_from').size().reset_index()
    drop_pendulum_users.columns = ['user', 'n']
    drop_pendulum_users = drop_pendulum_users[drop_pendulum_users['n']>1]['user'].tolist()
    el_sm = el_sm[el_sm['domain_from'].isin(drop_pendulum_users)]
    del drop_pendulum_users

    el = pd.concat([el_seo, el_sm]).dropna()
    
    labs = labeldf[['url', 'label']].copy()
    labs.columns = ['nodes', 'label']
    user_nodes = list(set(el_sm['domain_from']))
    user_node_labs = -2 # users are all -2s
    ulabs = pd.DataFrame({'nodes': user_nodes, 'label': user_node_labs})
    labs = pd.concat([labs, ulabs])
    labs = labs.drop_duplicates(subset = 'nodes')
    labs['type'] = np.where(labs['label']==-2, 'user', 'url')
    
    labs[labs['nodes'].isin(el.domain_from.tolist() + el.domain_to.tolist())].shape
    url_mapper = {url: i for i, url in enumerate(labs.nodes.tolist())}
    labs['id'] = labs['nodes'].map(url_mapper)
    el['source'] = el['domain_from'].map(url_mapper)
    el['target'] = el['domain_to'].map(url_mapper)
    el.drop(columns = ['domain_from', 'domain_to'], inplace=True)
    #labs(594449,4) #el (3682182)
    return labs, el, url_mapper

def import_users_only(edge_path, attr_path, sm_edge_path, label_path):
    """
    import all data- all entites without labels are assigned -2 in this function
    """
    el_seo = pd.read_csv(edge_path).dropna()[['domain_from', 'domain_to']]
    labeldf = pd.read_csv(label_path)
    # discretize labels with quantiles
    labeldf['label'] = pd.qcut(labeldf['pc1'], q=5, labels=[0, 1, 2, 3, 4])
    labeldf = labeldf[['domain', 'label']]
    labeldf.columns = ['url', 'label']
    
    labeldf = labeldf[labeldf.url.isin(el_seo.domain_to)]
    #### charitynavigator.org 
    
    urls_in_twitter = labeldf.url.unique().tolist()
    el_seo = el_seo[el_seo.domain_to.isin(urls_in_twitter)]

    #labeldf['label']= labeldf.label.astype(int).replace({6:1,5:1,4:0,3:0,2:0,1:0,-1:-1})
    # social media data
    el_sm = pd.read_csv(sm_edge_path).drop_duplicates().drop(columns = 'count')
    #el_sm.columns = ['domain_from', 'domain_to']
    #sites w/o labels
    other_sites = el_seo[~el_seo['domain_from'].isin(labeldf.url.unique().tolist())].domain_from.unique()
    other_sites = pd.DataFrame({'url':other_sites, 'label':-1})
    labeldf = pd.concat([labeldf, other_sites])
    
    el_sm.columns = ['domain_to', 'domain_from']
    # limit to only domains in social media data
    labeldf = labeldf[labeldf.url.isin(el_sm.domain_to.unique().tolist())]
    
    el_seo = el_seo[el_seo.domain_from.isin(labeldf.url)]
    el_seo = el_seo[el_seo.domain_to.isin(labeldf.url)]

    # if we wanted only labeled domains: 
    el_sm = el_sm[el_sm['domain_to'].isin(list(set(labeldf.url.tolist() + el_seo.domain_from.tolist())))]
    # drop users who've only linked to a single domain.
    
    drop_pendulum_urls = el_sm.groupby('domain_to').size().reset_index()
    drop_pendulum_urls.columns = ['user', 'n']
    drop_pendulums_urls = drop_pendulum_urls[drop_pendulum_urls['n']>1]['user'].tolist()
    el_sm = el_sm[el_sm['domain_to'].isin(drop_pendulums_urls)]
    del drop_pendulum_urls
    drop_pendulum_users = el_sm.groupby('domain_from').size().reset_index()
    drop_pendulum_users.columns = ['user', 'n']
    drop_pendulum_users = drop_pendulum_users[drop_pendulum_users['n']>1]['user'].tolist()
    el_sm = el_sm[el_sm['domain_from'].isin(drop_pendulum_users)]
    del drop_pendulum_users

    el = el_sm
    
    labs = labeldf[['url', 'label']].copy()
    labs.columns = ['nodes', 'label']
    user_nodes = list(set(el_sm['domain_from']))
    user_node_labs = -2 # users are all -2s
    ulabs = pd.DataFrame({'nodes': user_nodes, 'label': user_node_labs})
    labs = pd.concat([labs, ulabs])
    labs = labs.drop_duplicates(subset = 'nodes')
    labs['type'] = np.where(labs['label']==-2, 'user', 'url')
    
    #labs[labs['nodes'].isin(el.domain_from.tolist() + el.domain_to.tolist())].shape

    url_mapper = {url: i for i, url in enumerate(labs.nodes.tolist())}
    labs['id'] = labs['nodes'].map(url_mapper)
    el['source'] = el['domain_from'].map(url_mapper)
    el['target'] = el['domain_to'].map(url_mapper)
    el.drop(columns = ['domain_from', 'domain_to'], inplace=True)
    #labs(594449,4) #el (3682182)
    return labs, el, url_mapper


def to_pt_n2v_tensors(ulabs, el):
    train, inter_vt = train_test_split(ulabs, test_size=0.2, random_state=42, stratify=ulabs.label)
    test, val = train_test_split(inter_vt, test_size=0.5, random_state=42, stratify=inter_vt.label)
    train_mask = torch.tensor(train.id.values).long()
    val_mask = torch.tensor(val.id.values).long()
    test_mask = torch.tensor(test.id.values).long()

    y = torch.tensor(ulabs.label.values).long()
    edge_list = torch.tensor([el['source'].values, el['target'].values])
    
    data = Data(edge_index = edge_list, y=y, train_mask=train_mask, val_mask = val_mask, test_mask = test_mask)
    return data


def n2v_model(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Node2Vec(data.edge_index, embedding_dim=23, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    
    
    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc = model.test(z[data.train_mask], data.y[data.train_mask],
                         z[data.test_mask], data.y[data.test_mask],
                         max_iter=150)
        return acc

    for epoch in range(1, 30):
        loss = train()
        acc = test()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

    @torch.no_grad()
    def plot_points(colors):
        model.eval()
        z = model(torch.arange(data.num_nodes, device=device))
        z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
        y = data.y.cpu().numpy()

        plt.figure(figsize=(8, 8))
        for i in range(3):
            plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
        plt.axis('off')
        plt.show()

    colors = [
        '#ffc0cb', '#bada55', '#008080'
    ]
    return model


def seo_only_clean(ulabs, el):
    ulabs = ulabs[ulabs['type'] == 'url']
    el = el[el['source'].isin(ulabs.id.tolist())]
    el = el[el['target'].isin(ulabs.id.tolist())] 
    ulabs = ulabs[ulabs['id'].isin(el.source.tolist() + el.target.tolist())]
    #ulabs = ulabs.rename(columns = {'id':'old_id'})
    #ulabs = ulabs.reset_index().rename(columns={'index':'id'})
    #temp_mapper = {idx:i for idx, i in enumerate(ulabs.old_id.tolist())}
    #el['source'] = el['source'].map(temp_mapper)
    #el['target'] = el['target'].map(temp_mapper)
    return ulabs, el


def gen_node2vec_embs(edge_path, attr_path, embedding_path, sm_edge_path, label_path, seo_only = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ulabs, el, url_mapper = import_seo_and_users(edge_path, attr_path, sm_edge_path, label_path)
    
    if seo_only:
        ulabs, el = seo_only_clean(ulabs, el)
        
    data = to_pt_n2v_tensors(ulabs, el)
    data.num_nodes = torch.tensor(len(data.y))
    model = n2v_model(data)
    z = model(torch.arange(data.num_nodes, device=device))
    embeddings = z.detach().cpu().numpy()
    if seo_only:
        np.save(seo_only_embedding_path, embeddings)
    else:
        np.save(embedding_path, embeddings)
    return ulabs, el, data, url_mapper

#(571876, 23)

def node_masker_n2v(edge_path, attr_path, embedding_path, sm_edge_path, label_path, seo_only = False):
    ulabs, el, url_mapper = import_seo_and_users(edge_path, attr_path, sm_edge_path, label_path)
    if seo_only:
        ulabs, el = seo_only_clean(ulabs, el)
    data = to_pt_n2v_tensors(ulabs, el)
    
    user = ulabs[ulabs.label == -2]
    # can only calculate loss function on of the 1432 nodes w/ labels
    # need to explicitly feed those nodes as masks to the loaders
    otm = data.train_mask.numpy()
    otv = data.val_mask.numpy()
    ott = data.test_mask.numpy()
    unmasked = ulabs[ulabs.label >= 0].id.values
    
    train_mask = torch.tensor(otm[np.isin(otm, unmasked)]).long()
    val_mask = torch.tensor(otv[np.isin(otv, unmasked)]).long()
    test_mask = torch.tensor(ott[np.isin(ott, unmasked)]).long()
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    embeddings = np.load(embedding_path)
    X = torch.FloatTensor(embeddings)
    data.x = X
    # sample neighbors - we use this for minibatching
    train_loader = NeighborLoader(data, input_nodes=(data.train_mask),
                                num_neighbors=[25, 15], batch_size=64, shuffle=True,
                                num_workers=0)
    valid_loader = NeighborLoader(data, input_nodes = (data.val_mask),
                            num_neighbors=[25, 15], batch_size=64, shuffle=False,
                            num_workers=0)
    test_loader = NeighborLoader(data, input_nodes = (data.test_mask),
                            num_neighbors=[25, 15], batch_size=64, shuffle=False,
                            num_workers=0)
    return train_loader, valid_loader, test_loader    


def node_masker_n2v_curriculum(edge_path, attr_path, embedding_path, sm_edge_path, label_path, binary_labels = True, seo_only = False, users_only = False):
    if users_only:
        ulabs, el, url_mapper = import_users_only(edge_path, attr_path, sm_edge_path, label_path)
    else:
        ulabs, el, url_mapper = import_seo_and_users(edge_path, attr_path, sm_edge_path, label_path)
    binary_mapping = {-1:-1, -2:-2, 0:0, 1:0, 2:1, 3:1, 4:1}
    ulabs['orig_label'] = ulabs['label']
    if binary_labels:
        ulabs['label'] = ulabs.orig_label.map(binary_mapping)
    if seo_only:
        ulabs, el = seo_only_clean(ulabs, el)
    data = to_pt_n2v_tensors(ulabs, el)
    # need to explicitly feed labeled nodes as masks to the loaders
    otm = data.train_mask.numpy()
    otv = data.val_mask.numpy()
    ott = data.test_mask.numpy()
    labeled_nodes = ulabs[ulabs['label'] > -1]
    #unmasked = torch.tensor(ulabs.id.values)
    
    ############ CHANGE TO [0,4] ##################
    rel_tiles = ulabs[ulabs.label==1].copy().id.values
    urel_tiles = ulabs[ulabs.label==0].copy().id.values
    
    rel_tiles = np.array_split(rel_tiles, 4)
    urel_tiles = np.array_split(urel_tiles, 4)
    # they're already ordered by reliability, so can bucket by order
    
    batch_0 = np.concatenate([rel_tiles[0], urel_tiles[3]])
    batch_1 = np.concatenate([rel_tiles[1], urel_tiles[2]])
    batch_2 = np.concatenate([rel_tiles[2], urel_tiles[1]])
    batch_3 = np.concatenate([rel_tiles[3], urel_tiles[0]])

    #batch by ease
    #easy_unmasked = torch.tensor(easy_extremes.id.values)
    #hard_unmasked = torch.tensor(ulabs[ulabs['orig_label'].isin([1,2,3])].id.values)
    b0_train_mask = torch.tensor(otm[np.isin(otm, batch_0)]).long()
    b1_train_mask = torch.tensor(otm[np.isin(otm, batch_1)]).long()
    b2_train_mask = torch.tensor(otm[np.isin(otm, batch_2)]).long()
    b3_train_mask = torch.tensor(otm[np.isin(otm, batch_3)]).long()
    
    
    val_mask = torch.tensor(otv[np.isin(otv, labeled_nodes.id.values)]).long()
    test_mask = torch.tensor(ott[np.isin(ott, labeled_nodes.id.values)]).long()
    
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.b0_mask = b0_train_mask
    data.b1_mask = b1_train_mask
    data.b2_mask = b2_train_mask
    data.b3_mask = b3_train_mask
    
    if seo_only:
        embeddings = torch.FloatTensor(np.load(seo_only_embedding_path))
    else:
        embeddings = torch.FloatTensor(np.load(embedding_path))
    data.x = embeddings
    # sample neighbors - we use this for minibatching
    train_loaders = []
    for mask in [data.b0_mask, data.b1_mask, data.b2_mask, data.b3_mask]:
        train_loaders.append(NeighborLoader(data, input_nodes=(mask),
                                    num_neighbors=[25, 15], batch_size=64, shuffle=True,
                                    num_workers=0))
    #train_loader = NeighborLoader(data, input_nodes=(data.b0_mask),
    #                                num_neighbors=[25, 15], batch_size=64, shuffle=True,
    #                                num_workers=0)
    valid_loader = NeighborLoader(data, input_nodes = (data.val_mask),
                            num_neighbors=[25, 15], batch_size=64, shuffle=False,
                            num_workers=0)
    test_loader = NeighborLoader(data, input_nodes = (data.test_mask),
                            num_neighbors=[25, 15], batch_size=64, shuffle=False,
                            num_workers=0)
    return train_loaders, valid_loader, test_loader 

#
#a,b,c = node_masker_n2v_curriculum(edge_path, attr_path, embedding_path, sm_edge_path, label_path, binary_labels = True, seo_only = False)

def curr_hetero_node_masker(edge_path, attr_path, embedding_path, sm_edge_path, label_path, binary_labels = True, user_attrs = True, inference = False):
    ulabs, el, url_mapper = import_seo_and_users(edge_path, attr_path, sm_edge_path, label_path)
    binary_mapping = {-1:-1, -2:-2, 0:0, 1:0, 2:1, 3:1, 4:1}
    ulabs['orig_label'] = ulabs['label']
    if binary_labels:
        ulabs['label'] = ulabs.orig_label.map(binary_mapping)
        
    # align nodes and attributes
    orig_attrs = pd.read_csv(attr_path)
    urls = ulabs[ulabs['label'] >= -1].copy()
    urlattrs = urls[['nodes', 'type']].merge(orig_attrs, how = 'left', right_on = 'url', left_on = 'nodes').dropna()
    urls = urls[urls['nodes'].isin(urlattrs.nodes.tolist())]
    url_hetmapper = {i:idx for idx, i in enumerate(urls.nodes.tolist())}
    urls['new_id'] = urls['nodes'].map(url_hetmapper)
    url_idmapper = urls.set_index('id').to_dict()['new_id']
    
    #####HERE#########
    dredge2domain = pd.read_csv('embeddings/dredge/dredge2domain.csv')
    dredge2domain['new_id'] = dredge2domain['domain'].map(url_hetmapper)
    dredge2domain = dredge2domain.dropna()
    dredge2domain['new_id'] = dredge2domain['new_id'].astype(int)
    
    
    urlattrs['id'] = urlattrs['url'].map(url_hetmapper)
    urlattrs = urlattrs.dropna()
    urlattrs['id'] = urlattrs['id'].astype(int)
    urlattrs = urlattrs.sort_values('id')
    urlattrs = urlattrs.drop(columns = ['url','id', 'type', 'nodes'])
    
    orig_attrs = np.log(urlattrs.values + 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(orig_attrs)
    
    # correct edge list indices
    seo_el = el[el.source.isin(urls['id'].tolist())].copy()
    seo_el['source_idx'] = seo_el['source'].map(url_idmapper)
    seo_el['target_idx'] = seo_el['target'].map(url_idmapper)
    seo_el = seo_el[['source_idx', 'target_idx']]

    # create masks on new ids
    train, inter_vt = train_test_split(urls, test_size=0.2, random_state=42, stratify=urls.label)
    test, val = train_test_split(inter_vt, test_size=0.5, random_state=42, stratify=inter_vt.label)
    train = train[train.label>=0]
    val = val[val.label>=0]
    test = test[test.label>=0]

    ############ Set up Curriculum ##################
    rel_tiles = urls[urls.label==1].copy().id.values
    urel_tiles = urls[urls.label==0].copy().id.values
    
    rel_tiles = np.array_split(rel_tiles, 4)
    urel_tiles = np.array_split(urel_tiles, 4)
    # they're already ordered by reliability, so can bucket by order
    
    batch_0 = np.concatenate([rel_tiles[0], urel_tiles[3]])
    batch_1 = np.concatenate([rel_tiles[1], urel_tiles[2]])
    batch_2 = np.concatenate([rel_tiles[2], urel_tiles[1]])
    batch_3 = np.concatenate([rel_tiles[3], urel_tiles[0]])

    #batch by ease
    #easy_unmasked = torch.tensor(easy_extremes.id.values)
    #hard_unmasked = torch.tensor(ulabs[ulabs['orig_label'].isin([1,2,3])].id.values)
    b0_mask = torch.tensor(train[np.isin(train.id.values, batch_0)].new_id.values).long()
    b1_mask = torch.tensor(train[np.isin(train.id.values, batch_1)].new_id.values).long()
    b2_mask = torch.tensor(train[np.isin(train.id.values, batch_2)].new_id.values).long()
    b3_mask = torch.tensor(train[np.isin(train.id.values, batch_3)].new_id.values).long()

    #val_mask = torch.tensor(otv[np.isin(otv, labeled_nodes.id.values)]).long()
    #test_mask = torch.tensor(ott[np.isin(ott, labeled_nodes.id.values)]).long()
    #train_mask = torch.tensor(train.new_id.values).long()
    val_mask = torch.tensor(val.new_id.values).long()
    test_mask = torch.tensor(test.new_id.values).long()
    unlabeled_mask = torch.tensor(urls[urls.label==-1].copy().new_id.values)

    # align user nodes and attributes
    users = ulabs[ulabs['type'] == 'user'].copy()
    ## 
    if user_attrs:
        user_x = np.load(user_text_embedding_path)
    else:
        embeddings = np.load(embedding_path)
        user_x = embeddings[users.id.values]
    users['new_id'] = list(range(users.shape[0]))
    user_idmapper = users.set_index('id').to_dict()['new_id']
    
    # re-index user edgelist
    user_el = el[el.source.isin(users.id.tolist())].copy()
    user_el['source_idx'] = user_el['source'].map(user_idmapper)
    user_el['target_idx'] = user_el['target'].map(url_idmapper)
    user_el = user_el.dropna()
    user_el['target_idx'] = user_el['target_idx'].astype(int)
    user_el = user_el[['source_idx', 'target_idx']]

    # create heterogeneous dataset
    hetdat = HeteroData()
    hetdat['websites'].x = torch.FloatTensor(x_scaled)
    hetdat['websites'].y = torch.tensor(urls.label.values).long()
    hetdat['websites'].unlabeled_mask = unlabeled_mask
    #hetdat['websites'].train_mask = train_mask
    hetdat['websites'].val_mask = val_mask
    hetdat['websites'].test_mask = test_mask
    hetdat['users'].x = torch.FloatTensor(user_x)
    hetdat[('websites', 'to', 'websites')].edge_index = torch.tensor([seo_el['source_idx'].values, seo_el['target_idx'].values]).long()
    hetdat[('users', 'to', 'websites')].edge_index = torch.tensor([user_el['source_idx'].values, user_el['target_idx'].values]).long()
    #hetdat[('websites', 'to', 'users')].edge_index = torch.tensor([user_el['target_idx'].values, user_el['source_idx'].values]).long()
    
    hetdat = T.ToUndirected()(hetdat)
    #hetdat = T.AddSelfLoops()(hetdat)
    #hetdat = T.NormalizeFeatures()(hetdat)
    train_loaders = []
    for mask in [b0_mask, b1_mask, b2_mask, b3_mask]:
        train_loaders.append(NeighborLoader(hetdat, input_nodes= ('websites', mask),
                                num_neighbors={key: [25, 15] for key in hetdat.edge_types},
                                batch_size=64, shuffle=True, num_workers=0))
    # sample neighbors - we use this for minibatching
    #train_loader = NeighborLoader(hetdat, input_nodes= ('websites', hetdat['websites'].train_mask),
    #                            num_neighbors={key: [25, 15] for key in hetdat.edge_types},
    #                            batch_size=64, shuffle=True, num_workers=0)
    valid_loader = NeighborLoader(hetdat, input_nodes = ('websites', hetdat['websites'].val_mask),
                            num_neighbors={key: [25, 15] for key in hetdat.edge_types}, batch_size=64, shuffle=False,
                            num_workers=0)
    test_loader = NeighborLoader(hetdat, input_nodes = ('websites', hetdat['websites'].test_mask),
                            num_neighbors={key: [25,15] for key in hetdat.edge_types}, batch_size=64, shuffle=False,
                            num_workers=0)
    unlabeled_loader = NeighborLoader(hetdat, input_nodes = ('websites', hetdat['websites'].unlabeled_mask),
                            num_neighbors={key: [25,15] for key in hetdat.edge_types}, batch_size=64, shuffle=False,
                            num_workers=0)
    
    return train_loaders, valid_loader, test_loader, unlabeled_loader


def seo_only_attributed_import(edge_path, attr_path, embedding_path, sm_edge_path, label_path):
    ulabs, el, url_mapper = import_seo_and_users(edge_path, attr_path, sm_edge_path, label_path)
    binary_mapping = {-1:-1, -2:-2, 0:0, 1:0, 2:1, 3:1, 4:1}
    ulabs['orig_label'] = ulabs['label']
    ulabs, el = seo_only_clean(ulabs, el)
    ulabs['label'] = ulabs.orig_label.map(binary_mapping)

    #data = to_pt_n2v_tensors(ulabs, el)

    # align nodes and attributes
    orig_attrs = pd.read_csv(attr_path)
    urls = ulabs[ulabs['label'] >= -1].copy()
    urlattrs = urls[['nodes', 'type']].merge(orig_attrs, how = 'left', right_on = 'url', left_on = 'nodes').dropna()
    urls = urls[urls['nodes'].isin(urlattrs.nodes.tolist())]
    url_hetmapper = {i:idx for idx, i in enumerate(urls.nodes.tolist())}
    
    
    
    urls['new_id'] = urls['nodes'].map(url_hetmapper)
    url_idmapper = urls.set_index('id').to_dict()['new_id']
    
    urlattrs['id'] = urlattrs['url'].map(url_hetmapper)
    urlattrs = urlattrs.dropna()
    urlattrs['id'] = urlattrs['id'].astype(int)
    urlattrs = urlattrs.sort_values('id')
    urlattrs = urlattrs.drop(columns = ['url','id', 'type', 'nodes'])
    
    orig_attrs = np.log(urlattrs.values + 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(orig_attrs)
    
    # correct edge list indices
    seo_el = el[el.source.isin(urls['id'].tolist())].copy()
    seo_el['source_idx'] = seo_el['source'].map(url_idmapper)
    seo_el['target_idx'] = seo_el['target'].map(url_idmapper)
    seo_el = seo_el[['source_idx', 'target_idx']]

    # create masks on new ids
    train, inter_vt = train_test_split(urls, test_size=0.2, random_state=42, stratify=urls.label)
    test, val = train_test_split(inter_vt, test_size=0.5, random_state=42, stratify=inter_vt.label)
    train = train[train.label>=0]
    val = val[val.label>=0]
    test = test[test.label>=0]

    ############ Set up Curriculum ##################
    rel_tiles = urls[urls.label==1].copy().id.values
    urel_tiles = urls[urls.label==0].copy().id.values
    
    rel_tiles = np.array_split(rel_tiles, 4)
    urel_tiles = np.array_split(urel_tiles, 4)
    # they're already ordered by reliability, so can bucket by order
    
    batch_0 = np.concatenate([rel_tiles[0], urel_tiles[3]])
    batch_1 = np.concatenate([rel_tiles[1], urel_tiles[2]])
    batch_2 = np.concatenate([rel_tiles[2], urel_tiles[1]])
    batch_3 = np.concatenate([rel_tiles[3], urel_tiles[0]])

    #batch by ease
    #easy_unmasked = torch.tensor(easy_extremes.id.values)
    #hard_unmasked = torch.tensor(ulabs[ulabs['orig_label'].isin([1,2,3])].id.values)
    b0_mask = torch.tensor(train[np.isin(train.id.values, batch_0)].new_id.values).long()
    b1_mask = torch.tensor(train[np.isin(train.id.values, batch_1)].new_id.values).long()
    b2_mask = torch.tensor(train[np.isin(train.id.values, batch_2)].new_id.values).long()
    b3_mask = torch.tensor(train[np.isin(train.id.values, batch_3)].new_id.values).long()

    #val_mask = torch.tensor(otv[np.isin(otv, labeled_nodes.id.values)]).long()
    #test_mask = torch.tensor(ott[np.isin(ott, labeled_nodes.id.values)]).long()
    #train_mask = torch.tensor(train.new_id.values).long()
    val_mask = torch.tensor(val.new_id.values).long()
    test_mask = torch.tensor(test.new_id.values).long()
    # sample neighbors - we use this for minibatching
    
    y = torch.tensor(urls.label.values).long()
    edge_list = torch.vstack([torch.tensor(seo_el['source_idx'].values).long(), 
                             torch.tensor(seo_el['target_idx'].values).long()])
    
    data = Data(edge_index = edge_list, y=y, x = torch.FloatTensor(x_scaled), val_mask = val_mask, test_mask = test_mask)
    
    train_loaders = []
    for mask in [b0_mask, b1_mask, b2_mask, b3_mask]:
        train_loaders.append(NeighborLoader(data, input_nodes=(mask),
                                    num_neighbors=[25, 15], batch_size=64, shuffle=True,
                                    num_workers=0))
    #train_loader = NeighborLoader(data, input_nodes=(data.b0_mask),
    #                                num_neighbors=[25, 15], batch_size=64, shuffle=True,
    #                                num_workers=0)
    valid_loader = NeighborLoader(data, input_nodes = (data.val_mask),
                            num_neighbors=[25, 15], batch_size=64, shuffle=False,
                            num_workers=0)
    test_loader = NeighborLoader(data, input_nodes = (data.test_mask),
                            num_neighbors=[25, 15], batch_size=64, shuffle=False,
                            num_workers=0)
    return train_loaders, valid_loader, test_loader 
# gen n2vec embeddings
#ulabs, el, data, url_mapper =gen_node2vec_embs(edge_path, attr_path, embedding_path, sm_edge_path, label_path)

def dredge_import(edge_path, attr_path, embedding_path, sm_edge_path, label_path, dredge_words = True, binary_labels = True, user_attrs = False, inference = False):
    ulabs, el, url_mapper = import_seo_and_users(edge_path, attr_path, sm_edge_path, label_path)
    binary_mapping = {-1:-1, -2:-2, 0:0, 1:0, 2:1, 3:1, 4:1}
    ulabs['orig_label'] = ulabs['label']
    if binary_labels:
        ulabs['label'] = ulabs.orig_label.map(binary_mapping)
        
    # align nodes and attributes
    orig_attrs = pd.read_csv(attr_path)
    urls = ulabs[ulabs['label'] >= -1].copy()
    urlattrs = urls[['nodes', 'type']].merge(orig_attrs, how = 'left', right_on = 'url', left_on = 'nodes').dropna()
    urls = urls[urls['nodes'].isin(urlattrs.nodes.tolist())]
    url_hetmapper = {i:idx for idx, i in enumerate(urls.nodes.tolist())}
    urls['new_id'] = urls['nodes'].map(url_hetmapper)
    url_idmapper = urls.set_index('id').to_dict()['new_id']
    
    urlattrs['id'] = urlattrs['url'].map(url_hetmapper)
    urlattrs = urlattrs.dropna()
    urlattrs['id'] = urlattrs['id'].astype(int)
    urlattrs = urlattrs.sort_values('id')
    urlattrs = urlattrs.drop(columns = ['url','id', 'type', 'nodes'])
    
    orig_attrs = np.log(urlattrs.values + 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(orig_attrs)
    
    # correct edge list indices
    seo_el = el[el.source.isin(urls['id'].tolist())].copy()
    seo_el['source_idx'] = seo_el['source'].map(url_idmapper)
    seo_el['target_idx'] = seo_el['target'].map(url_idmapper)
    seo_el = seo_el[['source_idx', 'target_idx']]

    # create masks on new ids
    train, inter_vt = train_test_split(urls, test_size=0.2, random_state=42, stratify=urls.label)
    test, val = train_test_split(inter_vt, test_size=0.5, random_state=42, stratify=inter_vt.label)
    train = train[train.label>=0]
    val = val[val.label>=0]
    test = test[test.label>=0]

    ############ Set up Curriculum ##################
    rel_tiles = urls[urls.label==1].copy().id.values
    urel_tiles = urls[urls.label==0].copy().id.values
    
    rel_tiles = np.array_split(rel_tiles, 4)
    urel_tiles = np.array_split(urel_tiles, 4)
    # they're already ordered by reliability, so can bucket by order
    
    batch_0 = np.concatenate([rel_tiles[0], urel_tiles[3]])
    batch_1 = np.concatenate([rel_tiles[1], urel_tiles[2]])
    batch_2 = np.concatenate([rel_tiles[2], urel_tiles[1]])
    batch_3 = np.concatenate([rel_tiles[3], urel_tiles[0]])

    #batch by ease
    #easy_unmasked = torch.tensor(easy_extremes.id.values)
    #hard_unmasked = torch.tensor(ulabs[ulabs['orig_label'].isin([1,2,3])].id.values)
    b0_mask = torch.tensor(train[np.isin(train.id.values, batch_0)].new_id.values).long()
    b1_mask = torch.tensor(train[np.isin(train.id.values, batch_1)].new_id.values).long()
    b2_mask = torch.tensor(train[np.isin(train.id.values, batch_2)].new_id.values).long()
    b3_mask = torch.tensor(train[np.isin(train.id.values, batch_3)].new_id.values).long()

    #val_mask = torch.tensor(otv[np.isin(otv, labeled_nodes.id.values)]).long()
    #test_mask = torch.tensor(ott[np.isin(ott, labeled_nodes.id.values)]).long()
    #train_mask = torch.tensor(train.new_id.values).long()
    val_mask = torch.tensor(val.new_id.values).long()
    test_mask = torch.tensor(test.new_id.values).long()
    unlabeled_mask = torch.tensor(urls[urls.label==-1].copy().new_id.values)

    # align user nodes and attributes
    users = ulabs[ulabs['type'] == 'user'].copy()
    ## 
    if user_attrs:
        embeddings = np.load(user_text_embedding_path)
        user_x = embeddings[users.id.values]
    else:
        embeddings = np.load(embedding_path)
        user_x = embeddings[users.id.values]
    users['new_id'] = list(range(users.shape[0]))
    user_idmapper = users.set_index('id').to_dict()['new_id']
    
    # re-index user edgelist
    user_el = el[el.source.isin(users.id.tolist())].copy()
    user_el['source_idx'] = user_el['source'].map(user_idmapper)
    user_el['target_idx'] = user_el['target'].map(url_idmapper)
    user_el = user_el.dropna().copy()
    user_el['target_idx'] = user_el['target_idx'].astype(int)
    user_el = user_el[['source_idx', 'target_idx']]
    
    ## Bring in dredge words
    dredge_domain_el = pd.read_csv('embeddings/dredge/dredge2domain.csv').dropna()
    dredge_domain_el['domain_id'] = dredge_domain_el['domain'].map(url_hetmapper)
    dredge_domain_el = dredge_domain_el.dropna()
    dredge_domain_el['domain_id'] = dredge_domain_el['domain_id'].astype(int)
    
    dredge_idmapper = dredge_domain_el.set_index('dredge_word')['dredge_id'].to_dict()
    dredge_domain_el = dredge_domain_el[['dredge_id', 'domain_id']]

    dredge_sm_el = pd.read_csv('embeddings/dredge/user_to_dredge.csv')
    dredge_sm_el['ulabs_id'] = dredge_sm_el['corrupt_idx'].map(url_mapper)
    dredge_sm_el = dredge_sm_el.dropna()
    dredge_sm_el['ulabs_id'] = dredge_sm_el['ulabs_id'].astype(int)
    dredge_sm_el['user_id'] = dredge_sm_el['ulabs_id'].map(user_idmapper)
    dredge_sm_el['dredge_id'] = dredge_sm_el['string_match'].map(dredge_idmapper)
    dredge_sm_el = dredge_sm_el[['dredge_id', 'user_id']]
    
    dredge_feats = np.load('embeddings/dredge/dredge_word_embeddings.npy')
    #dredge_urls = pd.read_csv('implicit_sd/implicit_data/serp_results_t10only.csv')
    

    # create heterogeneous dataset
    hetdat = HeteroData()
    hetdat['websites'].x = torch.FloatTensor(x_scaled)
    hetdat['websites'].y = torch.tensor(urls.label.values).long()
    hetdat['websites'].unlabeled_mask = unlabeled_mask
    #hetdat['websites'].train_mask = train_mask
    hetdat['websites'].val_mask = val_mask
    hetdat['websites'].test_mask = test_mask
    hetdat['users'].x = torch.FloatTensor(user_x)
    hetdat['dredge'].x = torch.FloatTensor(dredge_feats)
    hetdat[('websites', 'to', 'websites')].edge_index = torch.tensor([seo_el['source_idx'].values, seo_el['target_idx'].values]).long()
    hetdat[('users', 'to', 'websites')].edge_index = torch.tensor([user_el['source_idx'].values, user_el['target_idx'].values]).long()
    hetdat[('dredge', 'to', 'websites')].edge_index = torch.tensor([dredge_domain_el['dredge_id'].values, dredge_domain_el['domain_id'].values]).long()
    hetdat[('dredge', 'to', 'users')].edge_index = torch.tensor([dredge_sm_el['dredge_id'].values, dredge_sm_el['user_id'].values]).long()

    #hetdat[('websites', 'to', 'users')].edge_index = torch.tensor([user_el['target_idx'].values, user_el['source_idx'].values]).long()
    
    hetdat = T.ToUndirected()(hetdat)
    #hetdat = T.AddSelfLoops()(hetdat)
    #hetdat = T.NormalizeFeatures()(hetdat)
    train_loaders = []
    for mask in [b0_mask, b1_mask, b2_mask, b3_mask]:
        train_loaders.append(NeighborLoader(hetdat, input_nodes= ('websites', mask),
                                num_neighbors={key: [25, 15] for key in hetdat.edge_types},
                                batch_size=64, shuffle=True, num_workers=0))
    # sample neighbors - we use this for minibatching
    #train_loader = NeighborLoader(hetdat, input_nodes= ('websites', hetdat['websites'].train_mask),
    #                            num_neighbors={key: [25, 15] for key in hetdat.edge_types},
    #                            batch_size=64, shuffle=True, num_workers=0)
    valid_loader = NeighborLoader(hetdat, input_nodes = ('websites', hetdat['websites'].val_mask),
                            num_neighbors={key: [25, 15] for key in hetdat.edge_types}, batch_size=64, shuffle=False,
                            num_workers=0)
    test_loader = NeighborLoader(hetdat, input_nodes = ('websites', hetdat['websites'].test_mask),
                            num_neighbors={key: [25,15] for key in hetdat.edge_types}, batch_size=64, shuffle=False,
                            num_workers=0)
    unlabeled_loader = NeighborLoader(hetdat, input_nodes = ('websites', hetdat['websites'].unlabeled_mask),
                            num_neighbors={key: [25,15] for key in hetdat.edge_types}, batch_size=64, shuffle=False,
                            num_workers=0)
    
    return train_loaders, valid_loader, test_loader, unlabeled_loader


#serp_results_t10 = pd.read_csv('implicit_sd/implicit_data/serp_results_t10only.csv')
## group by query and domain and get counts
#serp_results_t10 = serp_results_t10[['qry', 'domain']]
#serp_results_t10['count'] = 1
#serp_results_t10 = serp_results_t10.groupby(['qry', 'domain']).count().reset_index()
#serp_results_t10.to_csv('serp_to_dredge.csv', index = False)
#dtokeep = serp_results_t10['domain'].value_counts().reset_index()
#dtkeep = dtokeep[dtokeep['count'] > 1]
#s2d = serp_results_t10[serp_results_t10['domain'].isin(dtkeep['domain'])]
#s2d.to_csv('serp_to_dredge.csv', index = False)

#embedding_path = 'weights/node2vec_new_test.npy'
#data, idx_mapper = get_node2vec_embs(embedding_path)
#train_loader, valid_loader, test_loader = get_sm_only_network(embedding_path, url_split_only = True)

# import torch_geometric.transforms as T
# from torch_geometric.datasets import OGB_MAG
# from torch_geometric.nn import SAGEConv, to_hetero
# from torch_geometric.datasets import IMDB



def partial_f1_import(edge_path, attr_path, sm_edge_path, label_path):
    """
    import all data- all entites without labels are assigned -2 in this function
    """
    el_seo = pd.read_csv(edge_path).dropna()[['domain_from', 'domain_to']]
    pf1df = pd.read_csv('out/politicalnews/domains.csv')
    pf1df['url'] = pf1df['url'].str.lower()
    pf1df.columns = ['domain', 'label']
    labeldf = pf1df
    el_seo = el_seo[el_seo.domain_to.isin(pf1df.domain.tolist())]
    #labeldf = pd.read_csv(label_path)
    # discretize labels with quantiles
    #labeldf['label'] = pd.qcut(labeldf['pc1'], q=5, labels=[0, 1, 2, 3, 4])
    #labeldf = labeldf.merge(pf1df, how = 'inner', left_on = 'domain', right_on = 'url')
    labeldf = labeldf[['domain', 'label']]
    labeldf.columns = ['url', 'label']
    
    #pf1df[~pf1df.url.isin(labeldf.url.tolist())]
    labeldf = labeldf[labeldf.url.isin(el_seo.domain_to)]
    #### charitynavigator.org 
    
    urls_in_twitter = labeldf.url.unique().tolist()
    el_seo = el_seo[el_seo.domain_to.isin(urls_in_twitter)]

    #labeldf['label']= labeldf.label.astype(int).replace({6:1,5:1,4:0,3:0,2:0,1:0,-1:-1})
    # social media data
    el_sm = pd.read_csv(sm_edge_path).drop_duplicates().drop(columns = 'count')
    
    
    #el_sm.columns = ['domain_from', 'domain_to']
    #sites w/o labels
    other_sites = el_seo[~el_seo['domain_from'].isin(labeldf.url.unique().tolist())].domain_from.unique()
    dredge_domains = pd.read_csv('embeddings/dredge/dredge2domain.csv').dropna()
    dredge_domains = dredge_domains[~dredge_domains['domain'].isin(labeldf.url.unique().tolist())]
    dredge_domains = dredge_domains[~dredge_domains['domain'].isin(other_sites)]
    dredge_urls = dredge_domains.domain.unique().tolist()
    other_sites = other_sites.tolist() + dredge_urls
    
    other_sites = pd.DataFrame({'url':other_sites, 'label':-1})
    labeldf = pd.concat([labeldf, other_sites])

    el_sm.columns = ['domain_to', 'domain_from']
        
    # if we wanted only labeled domains: 
    target_users = el_sm[el_sm['domain_to'].isin(list(set(labeldf.url.tolist() + el_seo.domain_from.tolist())))].domain_from.unique().tolist()
    el_sm = el_sm[el_sm['domain_from'].isin(target_users)]

    # drop users who've only linked to a single domain.
    
    drop_pendulum_urls = el_sm.groupby('domain_to').size().reset_index()
    drop_pendulum_urls.columns = ['user', 'n']
    drop_pendulums_urls = drop_pendulum_urls[drop_pendulum_urls['n']>1]['user'].tolist()
    el_sm = el_sm[el_sm['domain_to'].isin(drop_pendulums_urls)]
    del drop_pendulum_urls
    drop_pendulum_users = el_sm.groupby('domain_from').size().reset_index()
    drop_pendulum_users.columns = ['user', 'n']
    drop_pendulum_users = drop_pendulum_users[drop_pendulum_users['n']>1]['user'].tolist()
    el_sm = el_sm[el_sm['domain_from'].isin(drop_pendulum_users)]
    del drop_pendulum_users
    
    ulabs, el, url_mapper = import_seo_and_users(edge_path, attr_path, sm_edge_path, label_path)
    candidate_urls = ulabs[ulabs['label'] >= -1]
    el_sm = el_sm[el_sm.domain_to.isin(candidate_urls.nodes.tolist())]
    
    addl_urls = el_sm[['domain_to']].drop_duplicates()
    addl_urls = addl_urls[~addl_urls['domain_to'].isin(labeldf['url'])]
    addl_urls['label'] = -1
    addl_urls.columns = ['url', 'label']
    labeldf = pd.concat([labeldf, addl_urls])

    el = pd.concat([el_seo, el_sm]).dropna()
    labs = labeldf[['url', 'label']].copy()
    labs.columns = ['nodes', 'label']
    user_nodes = list(set(el_sm['domain_from']))
    user_node_labs = -2 # users are all -2s
    ulabs = pd.DataFrame({'nodes': user_nodes, 'label': user_node_labs})
    labs = pd.concat([labs, ulabs])
    labs = labs.drop_duplicates(subset = 'nodes')
    labs['type'] = np.where(labs['label']==-2, 'user', 'url')
    
    labs[labs['nodes'].isin(el.domain_from.tolist() + el.domain_to.tolist())].shape

    url_mapper = {url: i for i, url in enumerate(labs.nodes.tolist())}
    labs['id'] = labs['nodes'].map(url_mapper)
    el['source'] = el['domain_from'].map(url_mapper)
    el['target'] = el['domain_to'].map(url_mapper)
    el.drop(columns = ['domain_from', 'domain_to'], inplace=True)
    #labs(594449,4) #el (3682182)
    return labs, el, url_mapper



def curr_hetero_node_masker(edge_path, attr_path, embedding_path, sm_edge_path, label_path, binary_labels = True, user_attrs = True, inference = False):
    ulabs, el, url_mapper = import_seo_and_users(edge_path, attr_path, sm_edge_path, label_path)
    binary_mapping = {-1:-1, -2:-2, 0:0, 1:0, 2:1, 3:1, 4:1}
    ulabs['orig_label'] = ulabs['label']
    if binary_labels:
        ulabs['label'] = ulabs.orig_label.map(binary_mapping)
        
    # align nodes and attributes
    orig_attrs = pd.read_csv(attr_path)
    urls = ulabs[ulabs['label'] >= -1].copy()
    urlattrs = urls[['nodes', 'type']].merge(orig_attrs, how = 'left', right_on = 'url', left_on = 'nodes').dropna()
    urls = urls[urls['nodes'].isin(urlattrs.nodes.tolist())]
    url_hetmapper = {i:idx for idx, i in enumerate(urls.nodes.tolist())}
    urls['new_id'] = urls['nodes'].map(url_hetmapper)
    url_idmapper = urls.set_index('id').to_dict()['new_id']
    
    urlattrs['id'] = urlattrs['url'].map(url_hetmapper)
    urlattrs = urlattrs.dropna()
    urlattrs['id'] = urlattrs['id'].astype(int)
    urlattrs = urlattrs.sort_values('id')
    urlattrs = urlattrs.drop(columns = ['url','id', 'type', 'nodes'])
    
    orig_attrs = np.log(urlattrs.values + 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(orig_attrs)
    
    # correct edge list indices
    seo_el = el[el.source.isin(urls['id'].tolist())].copy()
    seo_el['source_idx'] = seo_el['source'].map(url_idmapper)
    seo_el['target_idx'] = seo_el['target'].map(url_idmapper)
    seo_el = seo_el[['source_idx', 'target_idx']]

    # create masks on new ids
    train, inter_vt = train_test_split(urls, test_size=0.2, random_state=42, stratify=urls.label)
    test, val = train_test_split(inter_vt, test_size=0.5, random_state=42, stratify=inter_vt.label)
    train = train[train.label>=0]
    val = val[val.label>=0]
    test = test[test.label>=0]

    ############ Set up Curriculum ##################
    rel_tiles = urls[urls.label==1].copy().id.values
    urel_tiles = urls[urls.label==0].copy().id.values
    
    rel_tiles = np.array_split(rel_tiles, 4)
    urel_tiles = np.array_split(urel_tiles, 4)
    # they're already ordered by reliability, so can bucket by order
    
    batch_0 = np.concatenate([rel_tiles[0], urel_tiles[3]])
    batch_1 = np.concatenate([rel_tiles[1], urel_tiles[2]])
    batch_2 = np.concatenate([rel_tiles[2], urel_tiles[1]])
    batch_3 = np.concatenate([rel_tiles[3], urel_tiles[0]])

    #batch by ease
    #easy_unmasked = torch.tensor(easy_extremes.id.values)
    #hard_unmasked = torch.tensor(ulabs[ulabs['orig_label'].isin([1,2,3])].id.values)
    b0_mask = torch.tensor(train[np.isin(train.id.values, batch_0)].new_id.values).long()
    b1_mask = torch.tensor(train[np.isin(train.id.values, batch_1)].new_id.values).long()
    b2_mask = torch.tensor(train[np.isin(train.id.values, batch_2)].new_id.values).long()
    b3_mask = torch.tensor(train[np.isin(train.id.values, batch_3)].new_id.values).long()

    #val_mask = torch.tensor(otv[np.isin(otv, labeled_nodes.id.values)]).long()
    #test_mask = torch.tensor(ott[np.isin(ott, labeled_nodes.id.values)]).long()
    #train_mask = torch.tensor(train.new_id.values).long()
    val_mask = torch.tensor(val.new_id.values).long()
    test_mask = torch.tensor(test.new_id.values).long()
    unlabeled_mask = torch.tensor(urls[urls.label==-1].copy().new_id.values)

    # align user nodes and attributes
    users = ulabs[ulabs['type'] == 'user'].copy()
    ## 
    if user_attrs:
        user_x = np.load(user_text_embedding_path)
    else:
        embeddings = np.load(embedding_path)
        user_x = embeddings[users.id.values]
    users['new_id'] = list(range(users.shape[0]))
    user_idmapper = users.set_index('id').to_dict()['new_id']
    
    # re-index user edgelist
    user_el = el[el.source.isin(users.id.tolist())].copy()
    user_el['source_idx'] = user_el['source'].map(user_idmapper)
    user_el['target_idx'] = user_el['target'].map(url_idmapper)
    user_el = user_el.dropna()
    user_el['target_idx'] = user_el['target_idx'].astype(int)
    user_el = user_el[['source_idx', 'target_idx']]



    # create heterogeneous dataset
    hetdat = HeteroData()
    hetdat['websites'].x = torch.FloatTensor(x_scaled)
    hetdat['websites'].y = torch.tensor(urls.label.values).long()
    hetdat['websites'].unlabeled_mask = unlabeled_mask
    #hetdat['websites'].train_mask = train_mask
    hetdat['websites'].val_mask = val_mask
    hetdat['websites'].test_mask = test_mask
    hetdat['users'].x = torch.FloatTensor(user_x)
    hetdat[('websites', 'to', 'websites')].edge_index = torch.tensor([seo_el['source_idx'].values, seo_el['target_idx'].values]).long()
    hetdat[('users', 'to', 'websites')].edge_index = torch.tensor([user_el['source_idx'].values, user_el['target_idx'].values]).long()
    #hetdat[('websites', 'to', 'users')].edge_index = torch.tensor([user_el['target_idx'].values, user_el['source_idx'].values]).long()
    
    hetdat = T.ToUndirected()(hetdat)
    #hetdat = T.AddSelfLoops()(hetdat)
    #hetdat = T.NormalizeFeatures()(hetdat)
    train_loaders = []
    for mask in [b0_mask, b1_mask, b2_mask, b3_mask]:
        train_loaders.append(NeighborLoader(hetdat, input_nodes= ('websites', mask),
                                num_neighbors={key: [25, 15] for key in hetdat.edge_types},
                                batch_size=64, shuffle=True, num_workers=0))
    # sample neighbors - we use this for minibatching
    #train_loader = NeighborLoader(hetdat, input_nodes= ('websites', hetdat['websites'].train_mask),
    #                            num_neighbors={key: [25, 15] for key in hetdat.edge_types},
    #                            batch_size=64, shuffle=True, num_workers=0)
    valid_loader = NeighborLoader(hetdat, input_nodes = ('websites', hetdat['websites'].val_mask),
                            num_neighbors={key: [25, 15] for key in hetdat.edge_types}, batch_size=64, shuffle=False,
                            num_workers=0)
    test_loader = NeighborLoader(hetdat, input_nodes = ('websites', hetdat['websites'].test_mask),
                            num_neighbors={key: [25,15] for key in hetdat.edge_types}, batch_size=64, shuffle=False,
                            num_workers=0)
    unlabeled_loader = NeighborLoader(hetdat, input_nodes = ('websites', hetdat['websites'].unlabeled_mask),
                            num_neighbors={key: [25,15] for key in hetdat.edge_types}, batch_size=64, shuffle=False,
                            num_workers=0)
    
    return train_loaders, valid_loader, test_loader, unlabeled_loader


def partial_f1_graph(edge_path, attr_path, embedding_path, sm_edge_path, label_path):
    ulabs, el, url_mapper = partial_f1_import(edge_path, attr_path, sm_edge_path, label_path)
    #binary_mapping = {-1:-1, -2:-2, 0:0, 1:0, 2:1, 3:1, 4:1}
    #ulabs['orig_label'] = ulabs['label']
    #ulabs, el = seo_only_clean(ulabs, el)
    #ulabs['label'] = ulabs.orig_label.map(binary_mapping)

    #data = to_pt_n2v_tensors(ulabs, el)

    # align nodes and attributes
    orig_attrs = pd.read_csv(attr_path)
    urls = ulabs[ulabs['label'] >= -1].copy()
    urlattrs = urls[['nodes', 'type']].merge(orig_attrs, how = 'left', right_on = 'url', left_on = 'nodes').dropna()
    urls = urls[urls['nodes'].isin(urlattrs.nodes.tolist())].copy()
    url_hetmapper = {i:idx for idx, i in enumerate(urls.nodes.tolist())}
    urls['new_id'] = urls['nodes'].map(url_hetmapper)
    url_idmapper = urls.set_index('id').to_dict()['new_id']
    
    urlattrs['id'] = urlattrs['url'].map(url_hetmapper)
    urlattrs = urlattrs.dropna()
    urlattrs['id'] = urlattrs['id'].astype(int)
    urlattrs = urlattrs.sort_values('id')
    urlattrs = urlattrs.drop(columns = ['url','id', 'type', 'nodes'])
    
    orig_attrs = np.log(urlattrs.values + 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(orig_attrs)
    
    # correct edge list indices
    seo_el = el[el.source.isin(urls['id'].tolist())].copy()
    seo_el['source_idx'] = seo_el['source'].map(url_idmapper)
    seo_el['target_idx'] = seo_el['target'].map(url_idmapper)
    seo_el = seo_el[['source_idx', 'target_idx']]

    # create masks on new ids
    #train, inter_vt = train_test_split(urls, test_size=0.2, random_state=42, stratify=urls.label)
    #test, val = train_test_split(inter_vt, test_size=0.5, random_state=42, stratify=inter_vt.label)
    #train = train[train.label>=0]
    #val = val[val.label>=0]
    #test = test[test.label>=0]

    ############ Set up Curriculum ##################
    #rel_tiles = urls[urls.label==1].copy().id.values
    #urel_tiles = urls[urls.label==0].copy().id.values
    
    #rel_tiles = np.array_split(rel_tiles, 4)
    #urel_tiles = np.array_split(urel_tiles, 4)
    # they're already ordered by reliability, so can bucket by order
    
    #batch_0 = np.concatenate([rel_tiles[0], urel_tiles[3]])
    #batch_1 = np.concatenate([rel_tiles[1], urel_tiles[2]])
    #batch_2 = np.concatenate([rel_tiles[2], urel_tiles[1]])
    #batch_3 = np.concatenate([rel_tiles[3], urel_tiles[0]])

    #batch by ease
    #easy_unmasked = torch.tensor(easy_extremes.id.values)
    #hard_unmasked = torch.tensor(ulabs[ulabs['orig_label'].isin([1,2,3])].id.values)
    #b0_mask = torch.tensor(train[np.isin(train.id.values, batch_0)].new_id.values).long()
    #b1_mask = torch.tensor(train[np.isin(train.id.values, batch_1)].new_id.values).long()
    #b2_mask = torch.tensor(train[np.isin(train.id.values, batch_2)].new_id.values).long()
    #b3_mask = torch.tensor(train[np.isin(train.id.values, batch_3)].new_id.values).long()

    #val_mask = torch.tensor(otv[np.isin(otv, labeled_nodes.id.values)]).long()
    #test_mask = torch.tensor(ott[np.isin(ott, labeled_nodes.id.values)]).long()
    #train_mask = torch.tensor(train.new_id.values).long()
    #val_mask = torch.tensor(val.new_id.values).long()
    #test_mask = torch.tensor(test.new_id.values).long()
    unlabeled_mask = torch.tensor(urls[urls.label==-1].copy().new_id.values)
    labeled_mask = torch.tensor(urls[urls.label>=0].copy().new_id.values)

    # align user nodes and attributes
    users = ulabs[ulabs['type'] == 'user'].copy()
    ## 

    embeddings = np.load(embedding_path)
    user_x = embeddings[users.id.values]
    users['new_id'] = list(range(users.shape[0]))
    user_idmapper = users.set_index('id').to_dict()['new_id']
    
    # re-index user edgelist
    user_el = el[el.source.isin(users.id.tolist())].copy()
    user_el['source_idx'] = user_el['source'].map(user_idmapper)
    user_el['target_idx'] = user_el['target'].map(url_idmapper)
    user_el = user_el.dropna()
    user_el['target_idx'] = user_el['target_idx'].astype(int)
    user_el = user_el[['source_idx', 'target_idx']]

    ## Bring in dredge words
    dredge_domain_el = pd.read_csv('embeddings/dredge/dredge2domain.csv').dropna()
    dredge_domain_el['domain_id'] = dredge_domain_el['domain'].map(url_hetmapper)
    dredge_domain_el = dredge_domain_el.dropna()
    dredge_domain_el['domain_id'] = dredge_domain_el['domain_id'].astype(int)
    
    dredge_idmapper = dredge_domain_el.set_index('dredge_word')['dredge_id'].to_dict()
    dredge_domain_el = dredge_domain_el[['dredge_id', 'domain_id']]

    dredge_sm_el = pd.read_csv('embeddings/dredge/user_to_dredge.csv')
    dredge_sm_el['ulabs_id'] = dredge_sm_el['corrupt_idx'].map(url_mapper)
    dredge_sm_el = dredge_sm_el.dropna()
    dredge_sm_el['ulabs_id'] = dredge_sm_el['ulabs_id'].astype(int)
    dredge_sm_el['user_id'] = dredge_sm_el['ulabs_id'].map(user_idmapper)
    dredge_sm_el['dredge_id'] = dredge_sm_el['string_match'].map(dredge_idmapper)
    dredge_sm_el = dredge_sm_el[['dredge_id', 'user_id']]
    
    dredge_feats = np.load('embeddings/dredge/dredge_word_embeddings.npy')
    #dredge_urls = pd.read_csv('implicit_sd/implicit_data/serp_results_t10only.csv')
    


    # create heterogeneous dataset

    hetdat = HeteroData()
    hetdat['websites'].x = torch.FloatTensor(x_scaled)
    hetdat['websites'].y = torch.tensor(urls.label.values).long()
    hetdat['websites'].unlabeled_mask = unlabeled_mask
    #hetdat['websites'].train_mask = train_mask
    hetdat['websites'].val_mask = val_mask
    hetdat['websites'].test_mask = test_mask
    hetdat['users'].x = torch.FloatTensor(user_x)
    hetdat['dredge'].x = torch.FloatTensor(dredge_feats)
    hetdat[('websites', 'to', 'websites')].edge_index = torch.tensor([seo_el['source_idx'].values, seo_el['target_idx'].values]).long()
    hetdat[('users', 'to', 'websites')].edge_index = torch.tensor([user_el['source_idx'].values, user_el['target_idx'].values]).long()
    hetdat[('dredge', 'to', 'websites')].edge_index = torch.tensor([dredge_domain_el['dredge_id'].values, dredge_domain_el['domain_id'].values]).long()
    hetdat[('dredge', 'to', 'users')].edge_index = torch.tensor([dredge_sm_el['dredge_id'].values, dredge_sm_el['user_id'].values]).long()

    #hetdat[('websites', 'to', 'users')].edge_index = torch.tensor([user_el['target_idx'].values, user_el['source_idx'].values]).long()
    
    hetdat = T.ToUndirected()(hetdat)
    #hetdat = T.AddSelfLoops()(hetdat)
    #hetdat = T.NormalizeFeatures()(hetdat)

    # sample neighbors - we use this for minibatching
    #train_loader = NeighborLoader(hetdat, input_nodes= ('websites', hetdat['websites'].train_mask),
    #                            num_neighbors={key: [25, 15] for key in hetdat.edge_types},
    #                            batch_size=64, shuffle=True, num_workers=0)
    labeled_loader = NeighborLoader(hetdat, input_nodes = ('websites', hetdat['websites'].labeled_mask),
                            num_neighbors={key: [25, 15] for key in hetdat.edge_types}, batch_size=64, shuffle=False,
                            num_workers=0)

    unlabeled_loader = NeighborLoader(hetdat, input_nodes = ('websites', hetdat['websites'].unlabeled_mask),
                            num_neighbors={key: [25,15] for key in hetdat.edge_types}, batch_size=64, shuffle=False,
                            num_workers=0)
    
    full = NeighborLoader(hetdat, input_nodes = ('websites'),
                            num_neighbors={key: [25,15] for key in hetdat.edge_types}, batch_size=64, shuffle=False,
                            num_workers=0)
    
    return labeled_loader, unlabeled_loader

