import pandas as pd
import numpy as np
import torch
import os

from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import KBinsDiscretizer
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score, accuracy_score

import torch
from torch.nn.functional import one_hot
import torch_explain as te
from torch_explain.nn.functional import l1_loss
from torch_explain.logic.nn import psi
from torch_explain.logic.metrics import test_explanation,complexity
from imblearn.over_sampling import SMOTE

from preprocessing import load_data
from extractingExplanations import convertToConceptNames

# loading data from preprocessing - Psi networks do not need target variable to be one hot encoded
x, y, datay, concept_names = load_data(onehot=False)

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

# Stratified K-fold cross validation (Code style inspired from https://github.com/pietrobarbiero/pytorch_explain/blob/master/experiments/elens/mimic.py)
# **** Start of reference code from source mentioned above **** 
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for split, (trainval_index, test_index) in enumerate(skf.split(x.cpu().detach().numpy(),
                                                               y.argmax(dim=1).cpu().detach().numpy())):
    print(f'Split [{split + 1}/{n_splits}]')
    x_trainval, x_test = torch.FloatTensor(x[trainval_index]), torch.FloatTensor(x[test_index])
    y_trainval, y_test = torch.FloatTensor(datay[trainval_index]), torch.FloatTensor(datay[test_index])
    x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.2, random_state=42)
    
    # **** End of reference code ****

    print("Shape of training, validation and testing sets before SMOTE :")
    print(f'{len(y_train)}/{len(y_val)}/{len(y_test)}')
    
    oversample = SMOTE()
    x_train, y_train = oversample.fit_resample(x_train, y_train)

    print("Shape of training, validation and testing sets after SMOTE :")
    print(f'{len(y_train)}/{len(y_val)}/{len(y_test)}')

    x_train = torch.FloatTensor(x_train)
    y_train = one_hot(torch.tensor(y_train).to(torch.long))

    # Creating tensor datasets using training, testing and validation data
    train_data = TensorDataset(x_train, y_train)
    val_data = TensorDataset(x_val, y_val)
    test_data = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_data, batch_size=train_size)
    val_loader = DataLoader(val_data, batch_size=val_size)
    test_loader = DataLoader(test_data, batch_size=test_size)

    # creating Psi network layers
    layers = [
        torch.nn.Linear(x_train.shape[1], 20),
        torch.nn.Sigmoid(),
        torch.nn.Linear(20, 10),
        torch.nn.Sigmoid(),
        torch.nn.Linear(10, 5),
        torch.nn.Sigmoid(),
        torch.nn.Linear(5, 1),
        torch.nn.Sigmoid(),
    ]

    model = torch.nn.Sequential(* layers)

    # fitting and pruning the model
    optimizer = torch.optim.AdamW(model.parameters() , lr =0.001)
    loss_form = torch.nn.BCELoss()
    model.train()
    for epoch in range(10001):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_form(y_pred, y) + 0.000001 * te.nn.functional.l1_loss(model)
        optimizer.step()
        model = te.nn.functional.prune_equal_fanin(model, epoch, prune_epoch=1000)
    
    # predicting on test dataset and obtaining f1 & accuracy score
    y_pred = torch.argmax(model(x_test), axis=1)
    f1 = f1_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))
    acc = accuracy_score(y_test, y_pred)
    print("f1 = ", f1)

    # extracting FOL explanations for target class "didNotSurvive"
    explanation = psi.explain_class(model, x_test)
    y1h = one_hot(y_test.squeeze().long())

    # testing explanations and obtaining explaination evalutaion metrics
    accuracy, preds = test_explanation(explanation, x_test, y1h, target_class=1)
    explanation_complexity = complexity(explanation)

    if explanation == "False":
        pass
    elif explanation == "True":
        pass
    else:
        # converting explanations from feature numbers to concept names based explanations using Helper class
        expl = convertToConceptNames(explanation, concept_names)
        explanation_fidelity = accuracy_score(y_test.eq(1), preds) #obtaining explanation fidelity

        # creating dictionary with results of training model, explainer (evaluation metrics & explanations)
        dict = {'model_accuracy':[acc], 'f1': [f1], 'FOL_explanations': [explanation],  'Processed_explanations': [expl], 'explanations_f1': [accuracy], 'explanations_complexity': [explanation_complexity],
                'explanation_fidelity': [explanation_fidelity]}

        # checking current k-fold to ensure if new dataframe has to be created else append to existing results df
        if split == 0:
            df_result = pd.DataFrame.from_dict(dict)
        else:
            df_result = df_result.append(dict, ignore_index=True)

# Storing the results dataframe as a .csv file
base_dir = f'./results/psi'
df_result.to_csv(os.path.join(base_dir, 'results.csv'))

