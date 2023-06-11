import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import holidays

portugal_holidays = holidays.country_holidays('PT', subdiv='Ext')

def parse_time(x):
  dt = datetime.fromtimestamp(x["TIMESTAMP"])
  dt_tuple = dt.timetuple()
  holiday = 0
  if dt.date() in portugal_holidays:
    holiday = 2
  elif dt.date() + timedelta(days=1) in portugal_holidays:
    holiday = 1
  return (dt.hour*4 + dt.minute//15), dt.weekday(), dt.date().isocalendar().week - 1, holiday

raw_train = pd.read_csv("kaggle_data/encoding.csv")
test_df = pd.read_csv('kaggle_data/test_public.csv')

raw_train = raw_train[raw_train['IRREGULAR'] != True]
raw_train = raw_train.drop('IRREGULAR', axis=1)

test_df[["QTRHR", "WK", "WKYR", "HOLIDAY"]] = test_df[["TIMESTAMP"]].apply(parse_time, axis=1, result_type="expand")
test_df = test_df.drop(['MISSING_DATA', 'DAY_TYPE', 'TIMESTAMP'], axis=1)


test_df['ORIGIN_CALL'] = test_df['ORIGIN_CALL'].fillna(0)
test_df['ORIGIN_STAND'] = test_df['ORIGIN_STAND'].fillna(0)
raw_train['ORIGIN_CALL'] = raw_train['ORIGIN_CALL'].fillna(0)
raw_train['ORIGIN_STAND'] = raw_train['ORIGIN_STAND'].fillna(0)

label_encoder = LabelEncoder()

OC = sorted(raw_train['ORIGIN_CALL'].astype(int).unique())
label_encoder.fit(OC)
test_df['ORIGIN_CALL'] = test_df['ORIGIN_CALL'].apply(lambda x: 0 if x not in label_encoder.classes_ else x)
test_df['ORIGIN_CALL'] = label_encoder.transform(test_df['ORIGIN_CALL'].astype(int))


label_encoder.fit(range(64))
test_df['ORIGIN_STAND'] = label_encoder.transform(test_df['ORIGIN_STAND'].astype(int))

TI = sorted(raw_train['TAXI_ID'].astype(int).unique())
TI.insert(0, 0)
label_encoder.fit(TI)
test_df['TAXI_ID'] = test_df['TAXI_ID'].apply(lambda x: 0 if x not in label_encoder.classes_ else x)
test_df['TAXI_ID'] = label_encoder.transform(test_df['TAXI_ID'].astype(int))


print(test_df)




class MyModel(nn.Module):
    def __init__(self, embed):
        super(MyModel, self).__init__()

        self.embeddings = nn.ModuleList(
            [nn.Embedding(in_dim, out_dim) for in_dim, out_dim in embed]
        )
        
        em_dim = sum(embed.embedding_dim for embed in self.embeddings)
        self.linear1 = nn.Linear(em_dim, 128)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.linear2 = nn.Linear(128, 64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.linear3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = [embed(x[:, i]) for i, embed in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.batch_norm1(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.batch_norm2(x)
        x = self.linear3(x)
        return x.squeeze()

class EvalPandasDataset(Dataset):
    def __init__(self, dataframe, cat_cols):
        self.categorical = torch.tensor(dataframe[cat_cols].values, dtype=torch.int32)
        self.ids = dataframe['TRIP_ID']

    def __len__(self):
        return len(self.categorical)

    def __getitem__(self, index):
        return self.ids[index], self.categorical[index]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('high')

categorical_A = [
    'ORIGIN_CALL',
    'TAXI_ID',
    'QTRHR',
    'WK',
    'WKYR',
    'HOLIDAY'
]
embedding_dim_A = [
    (56481, 50),
    (443, 50),
    (96, 48),
    (7, 4),
    (52, 26),
    (3, 2)
]
categorical_B = [
    'ORIGIN_STAND',
    'TAXI_ID',
    'QTRHR',
    'WK',
    'WKYR',
    'HOLIDAY'
]
embedding_dim_B = [
    (64, 32),
    (443, 50),
    (96, 48),
    (7, 4),
    (52, 26),
    (3, 2)
]
categorical_C = [
    'TAXI_ID',
    'QTRHR',
    'WK',
    'WKYR',
    'HOLIDAY'
]
embedding_dim_C = [
    (443, 50),
    (96, 48),
    (7, 4),
    (52, 26),
    (3, 2)
]

test = test_df
A_data_test = test[test['CALL_TYPE'] == 'A'].reset_index()
B_data_test = test[test['CALL_TYPE'] == 'B'].reset_index()
C_data_test = test[test['CALL_TYPE'] == 'C'].reset_index()
A_test = EvalPandasDataset(A_data_test, categorical_A)
B_test = EvalPandasDataset(B_data_test, categorical_B)
C_test = EvalPandasDataset(C_data_test, categorical_C)
A_test_dataloader = DataLoader(A_test, batch_size=64, shuffle=False, num_workers=8)
B_test_dataloader = DataLoader(B_test, batch_size=64, shuffle=False, num_workers=8)
C_test_dataloader = DataLoader(C_test, batch_size=64, shuffle=False, num_workers=8)

model_A = MyModel(embedding_dim_A).to(device)
model_A.load_state_dict(torch.load('model_weights/A.pt'))
model_A.eval()
create_array = True
with torch.no_grad(), tqdm(A_test_dataloader, desc=f"A") as progress:
    for id, inputs in progress:
        inputs = inputs.to(device)
        outputs = model_A(inputs)

        if create_array:
            ids = id
            score = outputs
            create_array = False
        else:
            ids = ids + id
            score = torch.cat((score, outputs))

model_B = MyModel(embedding_dim_B).to(device)
model_B.load_state_dict(torch.load('model_weights/B.pt'))
model_B.eval()
with torch.no_grad(), tqdm(B_test_dataloader, desc=f"B") as progress:
    for id, inputs in progress:
        inputs = inputs.to(device)
        outputs = model_B(inputs)

        ids = ids + id
        score = torch.cat((score, outputs))

model_C = MyModel(embedding_dim_C).to(device)
model_C.load_state_dict(torch.load('model_weights/C.pt'))
model_C.eval()
with torch.no_grad(), tqdm(C_test_dataloader, desc=f"C") as progress:
    for id, inputs in progress:
        inputs = inputs.to(device)
        outputs = model_C(inputs)

        ids = ids + id
        score = torch.cat((score, outputs))
        
df = pd.DataFrame({'TRIP_ID': ids, 'TRAVEL_TIME': score.cpu()})
def extract_id(value):
    return int(value[1:])
df_sorted = df.iloc[df['TRIP_ID'].map(extract_id).argsort()]

print(df_sorted)

df_sorted.to_csv('submission.csv', index=False)