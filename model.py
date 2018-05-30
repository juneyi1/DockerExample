import pandas as pd
import numpy as np
import pickle

from string import digits, punctuation
import xml.etree.ElementTree as ET

from nltk import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from sklearn.pipeline import make_pipeline 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report

from icd_class import Tokenization, MatrixConverter

# After exploration done in ICD9_classification done in ICD9_classification.ipynb,
# codes were moved here

# Get data into dataaframe
docs = ET.parse('./2007ChallengeTrainData.xml')
doc_list = docs.findall('doc')

ID, CMC_MAJORITY, CLINICAL_HISTORY, IMPRESSION = [], [], [], []
for doc in doc_list:
    ID.append(doc.get('id'))    
    codes = [ code.text for code in doc[0] if code.get('origin') == "CMC_MAJORITY"]
    CMC_MAJORITY.append(codes)    
    CLINICAL_HISTORY.append(doc[1][0].text)
    IMPRESSION.append(doc[1][1].text)
    
data = {'CLINICAL_HISTORY': CLINICAL_HISTORY, 
        'IMPRESSION':IMPRESSION, 
        'ICD9_CM':CMC_MAJORITY}
df = pd.DataFrame(data, index=ID)
df['CLINICAL_HISTORY_IMPRESSION'] = df['CLINICAL_HISTORY'] + ' ' + df['IMPRESSION']

# Get unique codes for columns
codes = []
for code in df['ICD9_CM']:
    codes = codes + code
unique_codes = set(codes)
for code in unique_codes:
    df['ICD9_'+code] = df['ICD9_CM'].apply(lambda x: 1 if code in x else 0)
y_columns = [column for column in df.columns if (column.startswith('ICD9_') and column != 'ICD9_CM')]

# Prepare train, test split                
X = df['CLINICAL_HISTORY_IMPRESSION']
Y = df[y_columns]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create pipeline
tokenizer = Tokenization()
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,3), max_features=3000)
m2df = MatrixConverter()    # Convert a tfidf matrix to a dataframe
ss = StandardScaler()       #Standardize the dataframe
pca = PCA(n_components=782) # PCA the dataframe
ovr_mlp = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(15,15), 
                              solver='adam', 
                              alpha=1e-5,
                              learning_rate='adaptive',
                              activation='logistic',
                              max_iter=4000))
pipe = make_pipeline(tokenizer, tfidf, m2df, ss, pca, ovr_mlp)
# ovr_svc = OneVsRestClassifier(SVC(kernel='sigmoid', C=1.0, gamma=0.0008))
# pipe = make_pipeline(tokenizer, tfidf, m2df, ss, pca, ovr_svc)

pipe.fit(X_train, Y_train)
OVR_MLP_predict_train = pipe.predict(X_train)
OVR_MLP_predict_test = pipe.predict(X_test)
classification = classification_report(Y_test, OVR_MLP_predict_test)
print('train micro f1 score:', f1_score(Y_train, OVR_MLP_predict_train, average='micro'))
print('test micro f1 score:', f1_score(Y_test, OVR_MLP_predict_test, average='micro'))
print('test classification report:')
print(classification)

#Output 
pickle.dump(pipe, open("model_pipe.pkl", "wb"))
X_test.to_csv('X_test.csv', index=False)
Y_test.to_csv('Y_test.csv', index=False)
icd_codes = [code.replace('ICD9_','') for code in Y_test.columns]
pickle.dump(icd_codes, open("icd_codes.pkl", "wb"))