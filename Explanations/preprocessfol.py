import pandas as pd
import re
import numpy as np

# Helper function to process FOL and change symbols to strings and form a simple sentence
def preprocess_fol(df):
    df1 = df[['explanations0', 'explanations1']] # extracting explanation columns only 

    def replace_symbols(dfexplanation):
        sentence = []
        ans = []
        for sent in dfexplanation:
            sentence = sent.split(' ')
            for i in range(len(sentence)):
                if sentence[i] == '&':
                    sentence[i] = 'and'
                elif sentence[i] == "|":
                    sentence[i] = 'or'
                elif '~' == sentence[i][0]:
                    sentence[i] = "NOT "+ sentence[i][1:]
            ans.append(' '.join(sentence))
        return ans
    
    return replace_symbols(df1['explanations0']), replace_symbols(df1['explanations1'])