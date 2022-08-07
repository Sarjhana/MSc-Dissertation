# Certain excerpts have been adapted from https://github.com/pietrobarbiero/pytorch_explain/blob/master/experiments/elens/mimic.py
import pandas as pd
import numpy as np
import torch
import time
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from torch.nn.functional import one_hot
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import KBinsDiscretizer
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from torch_explain.models.explainer import Explainer
from torch_explain.logic.metrics import formula_consistency

from imblearn.over_sampling import SMOTE
from preprocessing import load_data

# loading data from preprocessing file
x, y, datay, concept_names = load_data(onehot=True)

# instantiating dataset in Tensor format and initialising variables
dataset = TensorDataset(x, y)
train_size = int(len(dataset) * 0.5)
val_size = (len(dataset) - train_size) // 2
test_size = len(dataset) - train_size - val_size
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_data, batch_size=train_size)
val_loader = DataLoader(val_data, batch_size=val_size)
test_loader = DataLoader(test_data, batch_size=test_size)
n_concepts = next(iter(train_loader))[0].shape[1]
n_classes = 2

seed_everything(42)

base_dir = f'./checkpoints'
os.makedirs(base_dir, exist_ok=True)

# Stratified K-fold cross validation (Code style inspired from https://github.com/pietrobarbiero/pytorch_explain/blob/master/experiments/elens/mimic.py)
# **** Start of reference code from source mentioned above **** 
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
results_list = []
feature_selection = []
explanations = {i: [] for i in range(n_classes)}
for split, (trainval_index, test_index) in enumerate(skf.split(x.cpu().detach().numpy(),
                                                               y.argmax(dim=1).cpu().detach().numpy())):
    print(f'Split [{split + 1}/{n_splits}]')
    x_trainval, x_test = torch.FloatTensor(x[trainval_index]), torch.FloatTensor(x[test_index])
    y_trainval, y_test = torch.FloatTensor(datay[trainval_index]), torch.FloatTensor(datay[test_index])
    x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.2, random_state=42)
    
    # **** End of reference code ****

    print("Shape of training, validation and testing sets before SMOTE :")
    print(f'{len(y_train)}/{len(y_val)}/{len(y_test)}')

    # Oversampling using SMOTE for training dataset
    oversample = SMOTE()
    x_train, y_train = oversample.fit_resample(x_train, y_train)

    y_train = one_hot(torch.tensor(y_train).to(torch.long))
    y_val = one_hot(torch.tensor(y_val).to(torch.long))
    y_test = one_hot(torch.tensor(y_test).to(torch.long))

    x_train = torch.FloatTensor(x_train)

    print("Shape of training, validation and testing sets after SMOTE :")
    print(f'{len(y_train)}/{len(y_val)}/{len(y_test)}')

    # Creating tensor datasets using training, testing and validation data
    train_data = TensorDataset(x_train, y_train)
    val_data = TensorDataset(x_val, y_val)
    test_data = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_data, batch_size=train_size)
    val_loader = DataLoader(val_data, batch_size=val_size)
    test_loader = DataLoader(test_data, batch_size=test_size)

    # Instantiating Trainer, Model (Entropy explainer) and checkpoint
    checkpoint_callback = ModelCheckpoint(dirpath=base_dir, monitor='val_loss', save_top_k=1)
    trainer = Trainer(max_epochs=500, gpus=0, auto_lr_find=True, deterministic=True,
                      check_val_every_n_epoch=1, default_root_dir=base_dir,
                      weights_save_path=base_dir, callbacks=[checkpoint_callback])
    model = Explainer(n_concepts=n_concepts, n_classes=n_classes, l1=1e-3, lr=0.01,
                      explainer_hidden=[10], temperature=0.80)

    # training interpretable classifier
    trainer.fit(model, train_loader, val_loader)

    print(f"Gamma: {model.model[0].concept_mask}")
    model.freeze()

    # Testing the model on test dataset and obtaining results
    model_results = trainer.test(model, test_loader)

    # obtaining f1 score for model training by testing on test set
    y_pred = torch.argmax(model(x_test), axis=1)
    y1 = torch.argmax(y_test, axis=1)
    f1 = f1_score(y1, y_pred, average = 'macro')
    print("f1 = ", f1)
    
    # extracting explanations from entropy explainer
    results, f = model.explain_class(val_loader, train_loader, test_loader,
                                     topk_explanations=10,
                                     concept_names=concept_names)

    # extracting model accuracy score
    results['model_accuracy'] = model_results[0]['test_acc_epoch']

     # creating dictionary with results of training model, explainer (evaluation metrics & explanations)
    dict = {'model_accuracy': [results['model_accuracy']],'f1': [f1], 'explanations0': [f[0]['explanation']], 'explanations0_f1': [f[0]['explanation_accuracy']], 'explanations0_fidelity': [f[0]['explanation_fidelity']], 'explanations0_complexity': [f[0]['explanation_complexity']],
            'explanations1': [f[1]['explanation']], 'explanations1_f1': [f[1]['explanation_accuracy']], 'explanations1_fidelity': [f[1]['explanation_fidelity']], 'explanations1_complexity': [f[1]['explanation_complexity']],
            }

    # checking current k-fold to ensure if new dataframe has to be created else append to existing results df
    if split == 0:
        df_result = pd.DataFrame.from_dict(dict)
    else:
        df_result = df_result.append(dict, ignore_index=True)

# Storing the results dataframe as a .csv file
base_dir = f'./results/entropy'
df_result.to_csv(os.path.join(base_dir, 'results.csv'))