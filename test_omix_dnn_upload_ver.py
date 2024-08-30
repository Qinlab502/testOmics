import math
import numpy as np
import pandas as pd
import os
import csv
import re
import requests
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import r2_score
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns


from torch.optim.lr_scheduler import LambdaLR


transcriptomics_data = pd.read_csv('./rna_accorwithprotein+-.csv', index_col=[0,1])
proteomics_data = pd.read_csv('./Proteins+-.csv', index_col=[0,1])
metabolomics_data = pd.read_csv('./Metabolomics.csv', index_col=[0])
metabolomics_data_transposed = metabolomics_data.transpose()
transcriptomics_data = transcriptomics_data.transpose()
proteomics_data = proteomics_data.transpose().iloc[1:-1]
proteomics_data.columns = proteomics_data.columns.set_levels(
    proteomics_data.columns.levels[0].str.upper(), level=0
)

X_rna = transcriptomics_data.values
X_pro = proteomics_data.values
X_meta = metabolomics_data_transposed.values

current_level_pro = proteomics_data.columns.get_level_values(0)
proteomics_data.columns = current_level_pro
current_level_trans = transcriptomics_data.columns.get_level_values(0)
transcriptomics_data.columns = current_level_trans

new_index=['Control-1', 'Control-2', 'Control-3', 'S7-1', 'S7-2', 'S7-3']
proteomics_data.index=new_index
metabolomics_data_transposed.index=new_index
print(transcriptomics_data.index)
print(transcriptomics_data.columns)
print(proteomics_data.index)
print(proteomics_data.columns)
print(metabolomics_data_transposed.index)
print(metabolomics_data_transposed.columns)


X = pd.concat([transcriptomics_data, proteomics_data, metabolomics_data_transposed], axis=1)
feature_name = X.columns.tolist()
feature_name_array = np.array(feature_name)

y = np.array([[1],
        [1],
        [1],
        [0.5],
        [0.5],
        [0.5]])
print(f'Shape of X: {X.shape}; Shape of y: {y.shape}')

all_feature_data = pd.DataFrame(X, columns=feature_name_array)
csv_path = './all_feature_data.csv'
all_feature_data.to_csv(csv_path, index=False)

class TestOmix(nn.Module):
    def __init__(self, In_Nodes, dropout_rate1, dropout_rate2, dim1, dim2):
        super(TestOmix, self).__init__()

        self.sc1 = nn.Linear(In_Nodes, dim1)
        self.sc2 = nn.Linear(dim1, dim2)
        self.sc3 = nn.Linear(dim2, 2, bias=True)
        self.sc4 = nn.Linear(2, 1, bias=True)

        init.xavier_uniform_(self.sc1.weight)
        init.xavier_uniform_(self.sc2.weight)
        init.xavier_uniform_(self.sc3.weight)
        init.xavier_uniform_(self.sc4.weight)

        self.dropout1 = nn.Dropout(dropout_rate1)
        self.dropout2 = nn.Dropout(dropout_rate2)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.sc1(x)
        x = self.tanh(x)
        x = self.dropout1(x)

        x = self.sc2(x)
        x = self.tanh(x)
        x = self.dropout2(x)

        x = self.sc3(x)
        x = self.tanh(x)
        x = self.sc4(x)

        return x
    
class CustomDataset(Dataset):

    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)

    def __getitem__(self, idx):
        return {'x': self.x[idx], 'y': self.y[idx]}


    def __len__(self):
        return len(self.x)

def reset_weights(m):
    """
    This function will reset model weights to avoid weight leakage.
    """
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        m.reset_parameters()

class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_loss):
        if self.best_score is None:
            self.best_score = current_loss
        elif current_loss > self.best_score + self.delta: 
            self.counter += 1
            if self.counter >= self.patience: 
                self.early_stop = True
        else:
            if current_loss < self.best_score - self.delta: 
                self.best_score = current_loss
            self.counter = 0 

def add_gaussian_noise(X, mean=0.0, std=0.01):
    noise = np.random.normal(mean, std, X.shape)
    return X + noise



def trainer(train_loader, test_loader, L2, lr, dropout_rate1, dropout_rate2, dim1, dim2):
    writer = SummaryWriter()  
    best_mae = float('inf')
    best_model_state = None

    model = TestOmix(In_Nodes=X.shape[1], dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2, dim1=dim1, dim2=dim2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.apply(reset_weights)

    criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=L2)

    global training_failures

    early_stopper = EarlyStopping(patience=100, delta=0.004)



    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            loss_record = []
            for batch in tqdm(train_loader, position=0, leave=True):
                inputs = batch['x'].to(device)
                labels = batch['y'].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.squeeze())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                loss_record.append(loss.detach().item())

            mean_train_loss = sum(loss_record) / len(loss_record)



            model.eval()
            total_valid_loss = 0.0
            total_samples = 0
            total_mae = 0.0

            for batch in test_loader:
                inputs, targets = batch['x'].to(device), batch['y'].to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                    valid_loss = F.smooth_l1_loss(outputs.squeeze(), targets.squeeze())
                    total_valid_loss += valid_loss.item() * targets.size(0)
                    mae = torch.abs(outputs - targets).sum().item()
                    total_mae += mae
                    total_samples += targets.size(0)

            mean_valid_loss = total_valid_loss / total_samples
            mean_mae = total_mae / total_samples

            print(f'Epoch [{epoch+1}/{num_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}, MAE: {mean_mae:.4f}')
            loss_dict = {'train': mean_train_loss, 'valid': mean_valid_loss}
            writer.add_scalars('Loss', loss_dict, epoch)
            writer.add_scalar('Mae/valid', mean_mae, epoch)


            if mean_mae < best_mae:
                best_mae = mean_mae
                best_model_state = model.state_dict()
                print(f"New Best MAE for this iteration: {best_mae}")

            early_stopper(mean_mae)
            if early_stopper.early_stop:
                print("Early stopping triggered!")
                break

    except Exception as e:
        print(f"Training failed with exception: {e}")
        training_failures += 1
        print(f"Number of training failures: {training_failures}")
        return None, float('inf')

    writer.close()
    return best_model_state, mean_train_loss, best_mae

dropout_rate1 = 0.3566765862692761
dropout_rate2 = 0.2928632703619797
dim1 = 474
dim2 = 176
lr = 0.0002228301757155853
L2 = 0.002039581520209592
num_epochs= 1500


transcriptomics_idx = (0, 597)
proteomics_idx = (597, 1069)
metabolomics_idx = (1069, 1562)


def split_and_scale(X, scaler, indices):
    X_subset = X.iloc[:, indices[0]:indices[1]]
    return scaler.fit_transform(X_subset)


scaler_transcriptomics = MinMaxScaler()
scaler_proteomics = MinMaxScaler()
scaler_metabolomics = MinMaxScaler()

X_removed = X.drop(['Control-1', 'S7-1'])

y_removed = np.delete(y, [0, 3])


print("X_removed shape:", X_removed.shape)
print("y_removed shape:", y_removed.shape)


best_mae_global = float('inf')
best_model_state_global = None
best_iteration = None
best_train_loss = None
training_failures = 0

loo = LeaveOneOut()

for i, (train_index, test_index) in enumerate(loo.split(X_removed)):

    X_train, X_test = X_removed.iloc[train_index], X_removed.iloc[test_index]
    y_train, y_test = y_removed[train_index], y_removed[test_index]


    X_train_transcriptomics_scaled = split_and_scale(X_train, scaler_transcriptomics, transcriptomics_idx)
    X_train_proteomics_scaled = split_and_scale(X_train, scaler_proteomics, proteomics_idx)
    X_train_metabolomics_scaled = split_and_scale(X_train, scaler_metabolomics, metabolomics_idx)


    X_train_scaled = np.concatenate(
        [X_train_transcriptomics_scaled, X_train_proteomics_scaled, X_train_metabolomics_scaled], axis=1
    )


    X_test_transcriptomics_scaled = split_and_scale(X_test, scaler_transcriptomics, transcriptomics_idx)
    X_test_proteomics_scaled = split_and_scale(X_test, scaler_proteomics, proteomics_idx)
    X_test_metabolomics_scaled = split_and_scale(X_test, scaler_metabolomics, metabolomics_idx)


    X_test_scaled = np.concatenate(
        [X_test_transcriptomics_scaled, X_test_proteomics_scaled, X_test_metabolomics_scaled], axis=1
    )


    X_train_noisy1 = add_gaussian_noise(X_train_scaled, mean=0.0, std=0.01)
    X_train_noisy2 = add_gaussian_noise(X_train_scaled, mean=0.0, std=0.01)
    X_train_noisy3 = add_gaussian_noise(X_train_scaled, mean=0.0, std=0.01)

    X_train_augmented = np.concatenate([X_train_scaled, X_train_noisy1, X_train_noisy2, X_train_noisy3], axis=0)
    y_train_augmented = np.concatenate([y_train, y_train, y_train, y_train], axis=0)

    train_dataset = CustomDataset(X_train_augmented, y_train_augmented)
    test_dataset = CustomDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("Train indices:", train_index, "Test indices:", test_index)
    print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)

    best_model_state, mean_train_loss, best_mae = trainer(train_loader, test_loader, L2, lr, dropout_rate1, dropout_rate2, dim1, dim2)

    if best_mae < best_mae_global:
        best_mae_global = best_mae
        best_model_state_global = best_model_state
        best_iteration = i
        best_train_loss = mean_train_loss

torch.save(best_model_state_global, './modelsDNN/global_best_model2.ckpt')

print(f"Best Model was from Iteration: {best_iteration}")
print(f"Mean Train Loss of Best Model: {best_train_loss}")
print(f"Mean Mae of Best Model: {best_mae_global}")

from sklearn.metrics import mean_absolute_error
import scipy.stats as stats

model = TestOmix(In_Nodes=X.shape[1], dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2, dim1=dim1, dim2=dim2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(torch.load("./modelsDNN/global_best_model.ckpt"))


X_selected = X.loc[['Control-1', 'S7-1']]


y_selected = y[[0, 3]]


X_selected_transcriptomics_scaled = split_and_scale(X_selected, scaler_transcriptomics, transcriptomics_idx)
X_selected_proteomics_scaled = split_and_scale(X_selected, scaler_proteomics, proteomics_idx)
X_selected_metabolomics_scaled = split_and_scale(X_selected, scaler_metabolomics, metabolomics_idx)


X_selected_scaled = np.concatenate(
    [X_selected_transcriptomics_scaled, X_selected_proteomics_scaled, X_selected_metabolomics_scaled], axis=1
)


eval_dataset = CustomDataset(X_selected_scaled, y_selected)
eval_loader = DataLoader(eval_dataset, batch_size=len(eval_dataset))

num_tests = 100
all_predictions = []
all_actuals = []

for _ in range(num_tests):
    predictions, actuals = [], []
    with torch.no_grad():
        for batch in eval_loader:
            inputs = batch['x'].to(device)
            targets = batch['y'].to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())


    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    all_predictions.append(predictions)
    all_actuals.append(actuals)


maes = [mean_absolute_error(a, p) for a, p in zip(all_actuals, all_predictions)]


mean_preds = np.mean(all_predictions, axis=0)
std_preds = np.std(all_predictions, axis=0)


confidence_interval = stats.norm.interval(0.95, loc=mean_preds, scale=std_preds / np.sqrt(num_tests))


plt.figure(figsize=(12, 8))


plt.style.use('ggplot')


for preds in all_predictions:
    plt.scatter(range(len(preds)), preds, alpha=0.2, c='dodgerblue', edgecolors='none', s=50, label='Individual Predictions')


plt.plot(all_actuals[0], label='Actual Values', c='black', linewidth=2)
plt.plot(mean_preds, label='Predicted Mean', c='crimson', linewidth=2)


plt.fill_between(range(len(all_actuals[0])), confidence_interval[0], confidence_interval[1], color='crimson', alpha=0.3)


plt.title('Actual vs. Predicted Values with 95% Confidence Interval', fontsize=18)
plt.xlabel('Sample Index', fontsize=14)
plt.ylabel('Value', fontsize=14)



start_offset = 0.95  
line_height = 0.04  


for i in range(len(mean_preds)):
    plt.text(1.05, start_offset - i * line_height * 2, f'Predicted Mean {i+1}: {mean_preds[i]:.2f}',
             horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)
    plt.text(1.05, start_offset - (i * line_height * 2 + line_height), f'95% CI {i+1}: [{confidence_interval[0][i]:.2f}, {confidence_interval[1][i]:.2f}]',
             horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)


plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)


plt.show()


r2_scores = [r2_score(actuals, predictions) for actuals, predictions in zip(all_actuals, all_predictions)]


mean_r2_score = np.mean(r2_scores)


std_r2_score = np.std(r2_scores)


plt.figure(figsize=(10, 6))


plt.plot(r2_scores, label='R² Score per Iteration', marker='o', linestyle='-', color='blue')


plt.axhline(y=mean_r2_score, color='green', linestyle='--', label=f'Mean R² Score ({mean_r2_score:.2f})')



confidence_interval = stats.norm.interval(0.95, loc=mean_r2_score, scale=std_r2_score / np.sqrt(len(r2_scores)))
plt.fill_between(range(len(r2_scores)), confidence_interval[0], confidence_interval[1], color='red', alpha=0.2, label='95% CI')


plt.legend()


plt.title('R² Score Over 100 iterations', fontsize=18)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('R² Score', fontsize=14)


plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(r2_scores)


plt.title('Distribution of R² Scores', fontsize=18)
plt.xlabel('R² Score', fontsize=14)


plt.show()

plt.figure(figsize=(8, 6))
sns.violinplot(y=r2_scores, palette='pastel')
plt.title('Distribution of R² Scores over 100 iterations', fontsize=18)
plt.ylabel('R² Score', fontsize=14)
plt.show()

from sklearn.manifold import MDS
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

X_transcriptomics_scaled = split_and_scale(X, scaler_transcriptomics, transcriptomics_idx)
X_proteomics_scaled = split_and_scale(X, scaler_proteomics, proteomics_idx)
X_metabolomics_scaled = split_and_scale(X, scaler_metabolomics, metabolomics_idx)


X_scaled = np.concatenate(
    [X_transcriptomics_scaled, X_proteomics_scaled, X_metabolomics_scaled], axis=1
)




all_sample_dataset = CustomDataset(X_scaled, y)


all_sample_loader = DataLoader(all_sample_dataset, batch_size=len(all_sample_dataset), shuffle=False)  

model = TestOmix(In_Nodes=X.shape[1], dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2, dim1=dim1, dim2=dim2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load("./modelsDNN/global_best_model.ckpt"))
model.to(device)

from lime import lime_tabular
import lime.lime_tabular
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error



X_transcriptomics_scaled = split_and_scale(X, scaler_transcriptomics, transcriptomics_idx)
X_proteomics_scaled = split_and_scale(X, scaler_proteomics, proteomics_idx)
X_metabolomics_scaled = split_and_scale(X, scaler_metabolomics, metabolomics_idx)


X_scaled = np.concatenate(
    [X_transcriptomics_scaled, X_proteomics_scaled, X_metabolomics_scaled], axis=1
)


explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_scaled,  
    mode='regression',  
    feature_names=feature_name_array  
)

model = TestOmix(In_Nodes=X.shape[1], dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2, dim1=dim1, dim2=dim2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load("./modelsDNN/global_best_model.ckpt"))

model.to(device)



def predict_fn(x):

    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        preds = model(x_tensor)
    return preds.cpu().numpy()

feature_importances = np.zeros((X.shape[0], X.shape[1]))  

for i in range(X.shape[0]):
    exp = explainer.explain_instance(X_scaled[i], predict_fn, num_features=X.shape[1])
    exp_map = {exp.domain_mapper.feature_names[j]: exp.local_exp[1][j][1] for j in range(X.shape[1])}
    feature_importances[i] = [exp_map[feature] for feature in feature_name_array]


feature_importances_df = pd.DataFrame(feature_importances, columns=feature_name_array)


output_file_path = './feature_importance_sample.csv'  
feature_importances_df.to_csv(output_file_path, index=False)

model = TestOmix(In_Nodes=X.shape[1], dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2, dim1=dim1, dim2=dim2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load("./modelsDNN/global_best_model.ckpt"))
model.to(device)
model.eval()


X_scaled_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)


model.eval()


with torch.no_grad():
    original_preds = model(X_scaled_tensor)
    original_score = mean_squared_error(y_tensor.cpu().numpy(), original_preds.cpu().numpy())


importances = []
for i in range(X.shape[1]):
    X_permuted = X_scaled_tensor.clone().detach()
    X_permuted[:, i] = torch.tensor(np.random.permutation(X_permuted[:, i].cpu().numpy()), device=device)
    with torch.no_grad():
        permuted_preds = model(X_permuted)
    permuted_score = mean_squared_error(y_tensor.cpu().numpy(), permuted_preds.cpu().numpy())
    importances.append(original_score - permuted_score)


feature_importances = {feature_name_array[i]: importances[i] for i in range(len(feature_name_array))}

plt.figure(figsize=(20, 10))


plt.bar(range(596), importances[:596], color='#1f77b4', label='Transcriptomics')


plt.bar(range(596, 1068), importances[596:1068], color='#2ca02c', label='Proteomics')


plt.bar(range(1068, X.shape[1]), importances[1068:], color='#d62728', label='Metabolomics')


plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Permutation Feature Importance')


plt.legend()

plt.show()


feature_importances_raw_df = pd.DataFrame({
    'Feature': feature_name_array,
    'Importance': importances
})


feature_importances_raw_df.to_csv('./features_importance.csv', index=False)

import matplotlib.pyplot as plt

file_path = './features_importance.csv'
features_data = pd.read_csv(file_path)


features_data['FeatureIndex'] = range(1, len(features_data) + 1)

colors = ['#1f77b4' if i <= 596 else '#2ca02c' if i <= 1068 else '#d62728' for i in features_data['FeatureIndex']]


plt.figure(figsize=(20, 10))


plt.scatter(features_data['FeatureIndex'], features_data['Importance'], c=colors, alpha=0.6, s=10)


plt.title('Feature Importance by Index', fontsize=20)
plt.xlabel('Feature Index', fontsize=16)
plt.ylabel('Importance', fontsize=16)


plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=10, label='Transcriptomics'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=10, label='Proteomics'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=10, label='Metabolomics')],
           title='Feature Groups')


plt.show()


file_path = './features_importance.csv'
features_importance_df = pd.read_csv(file_path)


def mark_feature_name(row_index):
    if 1 <= row_index <= 596:
        return '-trans'
    elif 597 <= row_index <= 1068:
        return '-pros'
    elif 1069 <= row_index <= 1562:
        return '-meta'
    else:
        return ''


features_importance_df['Feature'] = [
    f"{feature}{mark_feature_name(index + 1)}"
    for index, feature in enumerate(features_importance_df['Feature'])
]

features_importance_df.to_csv('./features_importance_classified.csv', index=False)



positive_df_tab = pd.read_csv('./features_importance_+40.csv', sep='\t')
negative_df_tab = pd.read_csv('./features_importance_-40.csv', sep='\t')


positive_features_tab = positive_df_tab['Feature']
positive_scores_tab = positive_df_tab['Importance']
negative_features_tab = negative_df_tab['Feature']
negative_scores_tab = negative_df_tab['Importance']


plt.figure(figsize=(20, 10))
colors_positive_tab = ['#1f77b4' if '-trans' in feature else '#2ca02c' if '-pros' in feature else '#d62728' for feature in positive_features_tab]
plt.bar(positive_features_tab, positive_scores_tab, color=colors_positive_tab, alpha=0.5, width=0.8)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Bar Chart of Top 40% Positive Feature Importance')
plt.xticks([])  
plt.show()


plt.figure(figsize=(20, 10))
colors_negative_tab_legend = ['#1f77b4' if '-trans' in feature else '#2ca02c' if '-pros' in feature else '#d62728' for feature in negative_features_tab[::-1]]
plt.bar(negative_features_tab, negative_scores_tab, color=colors_negative_tab_legend, alpha=0.5, width=0.8)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Bar Chart of Top 40% Negative Feature Importance')
plt.xticks([])  
plt.ylim(-max(positive_scores_tab), 0)  


plt.legend(handles=[plt.Rectangle((0,0),1,1, color=color) for color in ['#1f77b4', '#2ca02c', '#d62728']],
           labels=['Trans Features', 'Pros Features', 'Meta Features'])
plt.show()

import requests
import pandas as pd



file_path = '/content/feature_importance_meta_60.csv'
data = pd.read_csv(file_path, sep='\t')  

def query_kegg(gene):
    try:
        url = f"https://rest.kegg.jp/get/{gene}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.text  
        else:
            return None
    except Exception as e:
        return None


data['KEGG_Result'] = data.iloc[:, 0].apply(query_kegg)


output_file_path = './kegg_results.csv'  
data.to_csv(output_file_path, index=False)

print(f"结果保存到 {output_file_path}")


import xml.etree.ElementTree as ET

def query_ncbi(gene):
    try:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "gene",
            "term": gene,
            "retmode": "xml",
            "retmax": 1
        }

        search_response = requests.get(base_url, params=search_params)
        if search_response.status_code != 200:
            return "Search request failed"

        search_result = ET.fromstring(search_response.content)
        gene_id = search_result.find(".//Id").text if search_result.find(".//Id") is not None else None
        if gene_id is None:
            return "Gene ID not found"

        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        summary_params = {
            "db": "gene",
            "id": gene_id,
            "retmode": "xml"
        }

        summary_response = requests.get(summary_url, params=summary_params)
        if summary_response.status_code != 200:
            return "Summary request failed"

        summary_result = ET.fromstring(summary_response.content)
        summary = summary_result.find(".//Summary").text if summary_result.find(".//Summary") is not None else "No summary available"

        return summary
    except Exception as e:
        return str(e)


file_path = './features_importance_40.tsv'
data = pd.read_csv(file_path, sep='\t')  


data['NCBI_Result'] = data.iloc[:, 0].apply(query_ncbi)


output_file_path = './NCBI_results.csv'  
data.to_csv(output_file_path, index=False)

print(f"结果保存到 {output_file_path}")