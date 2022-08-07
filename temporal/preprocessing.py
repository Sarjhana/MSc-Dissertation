import pandas as pd
import numpy as np
import torch
import time
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from torch.nn.functional import one_hot
from sklearn.preprocessing import KBinsDiscretizer

from imblearn.over_sampling import SMOTE

def load_data(onehot):

    plotsDir = f'../plots/timeSeriesStatic'
    os.makedirs(plotsDir, exist_ok=True)

    # Load dataframe
    df = pd.read_csv('TimeSeries.csv')
    preprocessed_df = pd.DataFrame()

    # aggregation using the domain expert's input
    grouped = df.groupby('PatientID')

    preprocessed_df['Arterial_pH'] = grouped['Arterial pH'].max()
    preprocessed_df['AST'] = grouped['AST'].max()
    preprocessed_df['ALT'] = grouped['ALT'].max()
    preprocessed_df['WBC'] = grouped['WBC'].min()
    preprocessed_df['SVI'] = grouped['SVI'].min()
    preprocessed_df['SVR'] = grouped['SVR'].min()
    preprocessed_df['SVRI'] = grouped['SVRI'].min()
    preprocessed_df['Platelets'] = grouped['Platelets'].min()
    preprocessed_df['Haemoglobin'] = grouped['Hemoglobin'].min()
    preprocessed_df['Potassium'] = grouped['Potassium'].max()
    preprocessed_df['Sodium'] = grouped['Sodium'].max()
    preprocessed_df['Total_Bili'] = grouped['Total Bili'].max()
    preprocessed_df['Fibrinogen'] = grouped['Fibrinogen'].max()
    preprocessed_df['HeartRate'] = grouped['Heart Rate'].max()
    preprocessed_df['HeartRate'] = grouped['Heart Rate'].min()
    preprocessed_df['Glucose'] = grouped['Glucose'].min()
    preprocessed_df['Weight'] = grouped['Daily Weight'].mean()
    preprocessed_df['Albumin'] = grouped['Albumin'].min()
    preprocessed_df['Creatinine'] = grouped['Creatinine'].max()
    preprocessed_df['CaO2'] = grouped['CaO2'].min()
    preprocessed_df['INR'] = grouped['INR'].max()
    preprocessed_df['LDH'] = grouped['LDH'].max()
    preprocessed_df['ArterialBP_Diastolic'] = grouped['Arterial BP [Diastolic]'].min()
    preprocessed_df['ArterialBP_Systolic'] = grouped['Arterial BP [Systolic]'].min()
    preprocessed_df['Urea]'] = grouped['BUN'].max() * 0.357
    preprocessed_df['PTT'] = grouped['PTT'].max()
    preprocessed_df['Temperature'] = grouped['Temperature C'].max()
    preprocessed_df['CVP'] = grouped['CVP'].max()
    preprocessed_df['SpO2'] = grouped['SpO2'].min()
    preprocessed_df['SvO2'] = grouped['SvO2'].min()
    preprocessed_df['ArterialBP_Mean'] = grouped['Arterial BP Mean'].min()
    preprocessed_df['NBP_Mean'] = grouped['NBP Mean'].min()
    preprocessed_df['RespRateSpon'] = grouped['Resp Rate (Spont)'].max()

    # Missing values plot
    plt.figure()
    plt.plot(preprocessed_df.isnull().sum().sort_values(ascending = True), preprocessed_df.isnull().sum().sort_values(ascending = True).index)
    plt.xlabel('Count of missing values')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(plotsDir, 'missingValues-temporaldata.png'))

    # Summary of missing values
    print("Printing details of missing values")
    print(preprocessed_df.isnull().sum())

    # Extracting top 10 features + PatientID 
    features = list(preprocessed_df.isnull().sum().sort_values(ascending = True).head(10).index)
    col_names = list(preprocessed_df.columns)

    # Removing all other features from dataframe
    del_cols = [i for i in col_names if i not in features]
    preprocessed_df = preprocessed_df.drop(columns=del_cols)

    # encoding target variable
    preprocessed_df['Mortality'] = grouped['Mortality14Days'].apply(lambda x: 'didNotSurvive' if x.any() else 'Survived')

    # Data imbalance plot
    plt.figure()
    plt.bar(preprocessed_df['Mortality'].value_counts().index, preprocessed_df['Mortality'].value_counts())
    plt.tight_layout()
    plt.savefig(os.path.join(plotsDir, 'dataImbalance-temporaldata.png'))

    # Preparing y using LabelEncoding
    datay = preprocessed_df['Mortality']
    le = LabelEncoder()
    datay = le.fit_transform(datay)

    # dropping y variable from dataframe
    preprocessed_df = preprocessed_df.drop(columns=['Mortality'], axis=1)
    feature_names = list(preprocessed_df.columns)

    # Filling missing values using SimpleImputer
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = imp_mean.fit_transform(preprocessed_df)

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