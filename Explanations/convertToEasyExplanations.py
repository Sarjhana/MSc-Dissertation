from preprocessfol import preprocess_fol
import pandas as pd
import os

# Helper function to help store in a dataframe
def fol(explanations0, explanations1):
    key = ['explanations0', 'explanations1']
    new_dict = dict(zip(key, [explanations0, explanations1]))
    df_result = pd.DataFrame.from_dict(new_dict)
    return(df_result)

# Helper function to get processed FOL and store in a CSV from dataframe returned by fol()
def store_fol(df, fileName):
    explanations0, explanations1 = preprocess_fol(df)
    result_df = fol(explanations0, explanations1)
    base_dir = f'./explanation-results/'
    os.makedirs(base_dir, exist_ok=True)
    result_df.to_csv(os.path.join(base_dir, fileName))

# Convert breast cancer dataset's Entropy model generated explanations
df = pd.read_csv('/Users/sarjhana/Projects/Disso/breastcancer/results-breastcancer/resultsEntropy.csv')
store_fol(df, 'folBreastCancerEntropy.csv')

# Convert MIMIC static dataset's Entropy model generated explanations
df = pd.read_csv('/Users/sarjhana/Projects/Disso/mimicStatic/results/entropy/results.csv')
store_fol(df, 'folStaticEntropy.csv')

# Convert MIMIC time series dataset's Entropy model generated explanations
df = pd.read_csv('/Users/sarjhana/Projects/Disso/temporal/results/entropy/results.csv')
store_fol(df, 'folTimeSeriesEntropy.csv')




