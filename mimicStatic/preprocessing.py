import pandas as pd
import numpy as np
import torch
import time
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from torch.nn.functional import one_hot
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt

def load_data(onehot):

    # Load dataframe
    df = pd.read_csv('StaticData.csv')

    plotsDir = f'../plots/staticMIMIC'
    os.makedirs(plotsDir, exist_ok=True)

    df['deathperiod'] = df['deathperiod'].apply(lambda x: 'Survived' if x==-1 else 'didNotSurvive')
    datay = df['deathperiod'].to_numpy()

    # Data imbalance plot
    plt.figure()
    plt.bar(df['deathperiod'].value_counts().index, df['deathperiod'].value_counts())
    plt.tight_layout()
    plt.savefig(os.path.join(plotsDir, 'dataImbalance-staticMIMIC.png'))

    # Missing values plot
    plt.figure()
    plt.plot(df.isnull().sum().sort_values(ascending = True), df.isnull().sum().sort_values(ascending = True).index)
    plt.xlabel('Count of missing values')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(plotsDir, 'missingValues-staticMIMIC.png'))

    # Summary of missing values
    print("Printing details of missing values")
    print(df.isnull().sum())

    # Preparing y using LabelEncoding
    le = LabelEncoder()
    datay = le.fit_transform(datay)

    # dropping y variable from dataframe
    df = df.drop(columns=['deathperiod','PatientID'], axis=1)
    feature_names = list(df.columns)

    # Filling missing values using SimpleImputer
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = imp_mean.fit_transform(df)

    # Categorizing data using KBinsDiscretizer
    bin = KBinsDiscretizer(n_bins=3, encode='onehot-dense', strategy='uniform')
    datax = bin.fit_transform(data)

    x = torch.FloatTensor(datax)

    # if argument onhot is true then converting y to One-Hot encoding else converting datatype to float tensor
    if onehot == True:
        y = one_hot(torch.tensor(datay).to(torch.long)).to(torch.float)
    else:
        tens = torch.FloatTensor(datay)
        y = tens.view(len(datay), 1)

    # Preparing Concept Names from feature names (acc to bins)
    concept_names = []
    for feature in feature_names:
        concept_names.append(feature + '_LOW')
        concept_names.append(feature + '_NORMAL')
        concept_names.append(feature + '_HIGH')

    # Returns prepared x, one hot encoded y, without one hot encoded y, concept names
    return x, y, torch.FloatTensor(datay), concept_names
