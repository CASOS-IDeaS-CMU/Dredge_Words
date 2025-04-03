import pandas as pd
import networkx as nx
import numpy as np
from torch_geometric.loader import NeighborLoader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import torch
import random
random.seed(123)
np.random.seed(123)

def drop_corrupted_rows(links, labelled):
    """
    Drop rows with corrupted data- there were like 3 cryllic websites that the api couldn't ingest.
    They return nonsense- we'll delete them here.
    """
    a = links.domain_from.unique().tolist()
    b = links.domain_to.unique().tolist()
    uh = list(set(a +b))
    todrop_inlabeled = labelled[~labelled.domain_to.isin(uh)].domain_to
    labelled = labelled[~labelled.domain_to.isin(todrop_inlabeled)]
    
    # drop 3 and 4 labels - just to see how it affects model
    #labelled.label = labelled.label.astype(int)
    #labelled = labelled[labelled['label'] != 3]
    #labelled = labelled[labelled['label'] != 4]
    #labelled = (labelled.reset_index()).drop(columns = 'index')
    
    links = links[links.domain_from.isin(labelled.domain_to.tolist())]
    links = links[links.domain_to.isin(labelled.domain_to.tolist())]
    

    return links, labelled 



def data_only(labelled, links):
    """
    Generate train, val, and test masks and y tensor
    """
    targets = labelled[labelled['label'] > -1]
    ids = targets[['id', 'label']]
    X_train, X_v = train_test_split(ids, test_size=0.20, random_state=42, shuffle=True, stratify = ids.label)
    X_val, X_test = train_test_split(X_v, test_size=0.5, random_state=42, shuffle=True, stratify = X_v.label)

    train_mask = torch.tensor(X_train.id.values).long()
    val_mask = torch.tensor(X_val.id.values).long()
    test_mask = torch.tensor(X_test.id.values).long()

    y = torch.tensor(labelled.label.to_numpy()).long()

    # normalize feature matrix
    features = labelled.copy().drop(columns = ['url', 'id', 'label']).values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(features)
    x_scaled = torch.FloatTensor(x_scaled)

    edge_list = torch.tensor([links['source'].values, links['target'].values]).long()

    edge_weight = torch.FloatTensor(np.log([links['links'].values]))

    data = Data(x=x_scaled, edge_index = edge_list, edge_weight=edge_weight.reshape(-1), y=y, train_mask=train_mask, val_mask = val_mask, test_mask = test_mask)
    return data

def import_seo_links(data_input_path, links_input_path, labels_input_path):
    """
    import data
    """
    attributes = pd.read_csv(data_input_path).drop_duplicates(subset='domain_to')
    attributes['domain_to'] = attributes['domain_to'].str.lower()
    labels = pd.read_csv(labels_input_path)[['url', 'reliability']]
    labels.columns = ['domain_to', 'label']
    links = pd.read_csv(links_input_path).dropna()
    links = links[links.domain_from.isin(labels.domain_to)]

    if labels_input_path != "":
        if 'label' in attributes.columns:
            attributes.drop(columns='label', inplace=True)
        labelled = pd.merge(attributes, labels, how='left', on='domain_to').fillna(-1)
    links, labelled = drop_corrupted_rows(links, labelled)

    urls = labelled.domain_to.unique().tolist()
    url_mapper = {url: i for i, url in enumerate(urls)}
    labelled['id'] = labelled['domain_to'].map(url_mapper)
    links['source'] = links['domain_from'].map(url_mapper)
    links['target'] = links['domain_to'].map(url_mapper)

    links.drop(columns = ['domain_from', 'domain_to'], inplace=True)
    
    labelled.dropna(inplace=True)
    labelled['label'] = labelled.label.astype(int)#.replace(label_scheme)

    return labelled, links, url_mapper

def train_val_test_split(labelled, links):
    """
    Generate train, val, and test masks and y tensor
    """
    targets = labelled[labelled['label'] > -1]
    ids = targets[['id', 'label']]
    X_train, X_v = train_test_split(ids, test_size=0.20, random_state=42, shuffle=True, stratify = ids.label)
    X_val, X_test = train_test_split(X_v, test_size=0.5, random_state=42, shuffle=True, stratify = X_v.label)

    train_mask = torch.tensor(X_train.id.values).long()
    val_mask = torch.tensor(X_val.id.values).long()
    test_mask = torch.tensor(X_test.id.values).long()

    y = torch.tensor(labelled.label.to_numpy()).long()

    # normalize feature matrix
    features = labelled.copy().drop(columns = ['domain_to', 'id', 'label']).values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(np.log(features+1))
    #l2_x_norm = np.sqrt(np.einsum('ij,ij->j', x_scaled, x_scaled))
    #x_scaled = x_scaled/l2_x_norm
    x_scaled = torch.FloatTensor(x_scaled)
    
    # NOTE: comment for topN experiments
    #links[['so_ratio']] = links[['so_ratio']].replace(np.inf, 1)
    #link_attrs = links.loc[:, ['links','unique_pages','tb_ratio','so_ratio', 'e_tb_ratio','e_so_ratio','tp_ratio', 'sp_ratio']]
    # NOTE: uncomment for topN experiments
    # link_attrs = links.loc[:, ['links']]
    
    edge_list = torch.tensor([links['source'].values, links['target'].values]).long()
    edge_weight = torch.FloatTensor(np.log(links['links'].values+1))
    #edge_weight = torch.FloatTensor((links['domain_to_rating'].values+1) / 100)


    # edge_weight = torch.FloatTensor(np.log([x+1 for x in links['tb_ratio'].values])) # tb_ratio, so_ratio, tp_ratio, unique_pages

    data = Data(x=x_scaled, edge_index = edge_list, edge_weight=edge_weight, y=y, train_mask=train_mask, val_mask = val_mask, test_mask = test_mask)

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

