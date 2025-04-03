import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from gnns.seo_import import import_seo_links, train_val_test_split
from sklearn.metrics import  accuracy_score, f1_score
from torch_geometric.seed import seed_everything
from torch_geometric.nn import to_hetero
from gnns.model import GAT, GNN, HetGNN, HetGAT, HGT
from gnns.n2v2 import *
import random


NETWORK_TYPE = 'sm' #['links', 'sm'] 'sm' = social media users + links
edge_path = 'ahrefs/labeled/backlinks.csv'
sm_edge_path = 'edgelists/user_to_url.csv'
attr_path = 'ahrefs/labeled/attributes.csv'
label_path = 'domain_ratings.csv'
user_emb_path = 'embeddings/textv2.embeddings_550k.npy'
bad_drop = True
BINARY_LABELS = True
CURRICULUM = False
SEO_ONLY = False
PATIENCE = 50
TEXT_ATTRS = False
CURRICULUM_PATIENCE = 10
HETTRANS = False
if SEO_ONLY:
    embedding_path = 'embeddings/emb_seo_only.npy'
else:
    embedding_path = 'embeddings/emb_test.npy'


EPOCHS = 1000
# for balanced, lets make it a bit random
seed_everything(42)

# choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#device = 'cpu'
def init_params():
    with torch.no_grad():
        # Initialize lazy parameters via forwarding a single batch to the model:
        batch = next(iter(train_loaders[0]))
        batch = batch.to(device)
        model(batch.x_dict, batch.edge_index_dict)

def train(hettrans = True):
    """
    Trains the model
    """
    model.train()
    total_loss = total_correct = 0
    train_loader_len = 0
    possible_correct = 0
    predictions = []
    true_labs = []
    if CURRICULUM:
        if baby_steps.big_steps:
            curr_loader = list(train_loaders[0]) + list(train_loaders[1]) + list(train_loaders[2]) + list(train_loaders[3])
            np.random.shuffle(curr_loader)
        else:
            curr_loader = []
            for nloader in baby_steps.batch_n:
                curr_loader += list(train_loaders[nloader])
                np.random.shuffle(curr_loader)
    
    else:
        curr_loader = list(train_loaders[0]) + list(train_loaders[1]) + list(train_loaders[2]) + list(train_loaders[3])
        np.random.shuffle(curr_loader)
    
    for batch in tqdm(curr_loader):
    #for batch in tqdm(train_loader):
        batch_size = batch['websites'].batch_size
        optimizer.zero_grad()
        batch = batch.to(device)
        if hettrans:
            out = model(batch.x_dict, batch.edge_index_dict)[:batch_size]
        else:
            out = model(batch.x_dict, batch.edge_index_dict)['websites'][:batch_size]
        loss = loss_fn(out, batch['websites'].y[:batch_size])
        
        loss.backward()
        optimizer.step()
        
        pred = out.argmax(dim=-1)
        
        possible_correct += batch_size
        total_correct += int((pred == batch['websites'].y[:batch_size]).sum())
        
        total_loss += float(loss)
        train_loader_len += 1
        
        predictions.append(pred.cpu().numpy())
        true_labs.append(batch['websites'].y[:batch_size].cpu().numpy())
        
    train_loss = total_loss / train_loader_len
    #train_acc = total_correct / possible_correct
    train_acc = accuracy_score(np.concatenate(true_labs), np.concatenate(predictions))
    f1 = f1_score(np.concatenate(true_labs), np.concatenate(predictions), average = 'macro')

    return train_loss, train_acc, f1

def test(loader, hettrans = True):
    with torch.no_grad():
        model.eval()
        true_labs = []
        total_loss = 0
        train_loader_len = 0
        predictions = []
        true_labs = []
        for batch in tqdm(loader):
            batch = batch.to(device)
            batch_size = batch['websites'].batch_size
            if hettrans:
                out = model(batch.x_dict, batch.edge_index_dict)[:batch_size]
            else:
                out = model(batch.x_dict, batch.edge_index_dict)['websites'][:batch_size]
            batch_y = batch['websites'].y[:batch_size]

            # calculate validation loss
            loss = loss_fn(out, batch_y)
            pred = out.argmax(dim=-1)
                        
            total_loss += float(loss)
            train_loader_len += 1
            
            predictions.append(pred.cpu().numpy())
            true_labs.append(batch_y.cpu().numpy())
        
    valid_loss = total_loss / train_loader_len
    #valid_acc = total_correct / possible_correct           
    valid_acc = accuracy_score(np.concatenate(true_labs), np.concatenate(predictions))
    f1 = f1_score(np.concatenate(true_labs), np.concatenate(predictions), average = 'macro')

    return valid_loss, valid_acc, f1


class EarlyStopping():
    def __init__(self, patience=5, min_delta=0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.best_score = None
        self.best_epoch = 0
    def __call__(self, validation_loss, model):
        
        score = -validation_loss
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = 0
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), './out/het_gnn.pt')



class BabyStepsCurriculum():
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.big_steps = False
        self.best_score = None
        self.best_epoch = 0
        self.batch_n = [0]
        self.max_n = 4
        
    def __call__(self, validation_loss):
        score = -validation_loss
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = 0
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if len(self.batch_n) < self.max_n:
                    next_batch = [max(self.batch_n) + 1]
                    self.batch_n = self.batch_n + next_batch
                    if len(self.batch_n) == self.max_n:
                        self.big_steps = True
                        print('Baby grew up!')
                    self.counter = 0
                else:
                    pass
        else:
            self.best_score = score
            self.counter = 0


#a = hetdat['websites'].y[hetdat['websites'].train_mask]


if __name__ == '__main__':
    ## Run the script
    
    ## HOMOGENOUS LINKS ONLY RUNS (links-only):
    # creates network of 1,432 links/ backlinks
    #labelled, links, url_mapper = import_seo_links(data_input_path, links_input_path)
    #train_loader, valid_loader, test_loader = train_val_test_split(labelled, links)

    # for node2vec (only) homogenous runs
    #ulabs, el, url_mapper = import_seo_and_users(edge_path, attr_path)
    #data = to_pt_n2v_tensors(ulabs, el)
    #train_loader, valid_loader, test_loader= node_masker_n2v_hetweights(edge_path, attr_path, embedding_path)
    
    # for heterogeneous run:
    
    train_loaders, valid_loader, test_loader, unlabeled_loader = curr_hetero_node_masker(edge_path, attr_path, embedding_path, sm_edge_path, label_path, user_attrs = TEXT_ATTRS)
    
    # Change model here
    training_stats = []

    #m
    batch = next(iter(train_loaders[0]))
    
    
    macro_stats = []

    for trial in list(range(10)):
        if HETTRANS:
            model = HGT(hidden_channels=64, out_channels=2,
                    num_heads=2, num_layers=2, data = batch)
        else:
            model = GNN(512, 2)
            model = to_hetero(model, batch.metadata(), aggr='sum')

        model = model.to(device)
        init_params()

        #optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        loss_fn = torch.nn.CrossEntropyLoss()
        baby_steps = BabyStepsCurriculum(patience= CURRICULUM_PATIENCE, min_delta = 0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300, 2e-5)
        early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.001)
        for epoch in range(0, EPOCHS):

            tloss, tacc, tf1 = train(HETTRANS)
            #print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
            vloss, vacc, vf1 = test(valid_loader, hettrans=HETTRANS)
            _, test_accuracy, test_f1 = test(test_loader, hettrans=HETTRANS)
            scheduler.step()

            print(f'Epoch: {epoch}')
            print(f'Train: {tacc:.4f}, Val: {vacc:.4f}, T_loss: {tloss:.4f} V_loss: {vloss:.4f}')

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch + 1,
                    'Training Accuracy': tacc,
                    'Training Loss': tloss,
                    'Training F1': tf1,
                    'Valid. Loss': vloss,
                    'Valid. Accur.': vacc,
                    'Valid. F1': vf1,
                    'Test Accuracy' : test_accuracy,
                    'Test F1': test_f1
                }
            )
            baby_steps(vloss)        
            early_stopping(vloss, model)
            if early_stopping.early_stop:
                print("We are at epoch:", epoch)
                
                acc_csv = pd.DataFrame({'validation_accuracy' : training_stats[epoch-PATIENCE]['Valid. Accur.'],
                                        'training_accuracy': training_stats[epoch-PATIENCE]['Training Accuracy'],
                                        'test_accuracy': training_stats[epoch-PATIENCE]['Test Accuracy'],
                                        'train_f1': training_stats[epoch-PATIENCE]['Training F1'], 
                                        'val_f1':training_stats[epoch-PATIENCE]['Valid. F1'],
                                        'test_f1':training_stats[epoch-PATIENCE]['Test F1']}, index = [0])
                break
        
        if not early_stopping.early_stop:
            acc_csv = pd.DataFrame({'validation_accuracy' : training_stats[EPOCHS-1]['Valid. Accur.'],
                                    'training_accuracy': training_stats[EPOCHS-1]['Training Accuracy'],
                                    'test_accuracy': training_stats[EPOCHS-1]['Test Accuracy'],
                                    'train_f1': training_stats[EPOCHS-1]['Training F1'], 
                                    'val_f1':training_stats[EPOCHS-1]['Valid. F1'],
                                    'test_f1':training_stats[EPOCHS-1]['Test F1']}, index = [0])
        
        macro_stats.append(acc_csv)


final = pd.concat(macro_stats)
#final.to_csv('results/hetero.csv', index = False)
#final = pd.read_csv('results/hetero.csv')
final['test_accuracy'].mean()
final['test_accuracy'].std()
final['test_f1'].mean()
final['test_f1'].std()

#batch = next(iter(train_loaders[0])



def label_unlabeled_data():
    model = GNN(23, 2)
    model = to_hetero(model, batch.metadata(), aggr='sum')
    model.load_state_dict(torch.load('out/het_gnn.pt'))
    model = model.to(device)

    with torch.no_grad():
        model.eval()
        predictions = []
        confidence = []
        for batch in tqdm(full):
            batch = batch.to(device)
            batch_size = batch['websites'].batch_size
            out = model(batch.x_dict, batch.edge_index_dict)['websites'][:batch_size]
            confidence_score = torch.max(torch.exp(out), axis = 1).values.cpu().detach().numpy()
            pred = out.argmax(dim=-1)

            predictions.append(pred.cpu().numpy())
            confidence.append(confidence_score)

    predictions = np.concatenate(predictions)
    confidence = np.concatenate(confidence)
    
    #urls['preds'] = predictions
    #urls['confidence'] = confidence
    
    #urls.to_csv('polit_gnn_preds.csv', index = False)
    
    df = pd.read_csv('unlabeled.csv')
    df['preds'] = predictions
    df['confidence'] = confidence

    df = df.sort_values('confidence', ascending = False)
    df = df[df['preds'] == 0]
    df.to_csv('unreliable_preds_full.csv', index = False)
    df.head(200).to_csv('unreliable_top_200.csv', index = False)
    






