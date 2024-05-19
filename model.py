import pandas as pd
import pickle
column_name =["exp","id","time","query","id-name","comments"]
df=pd.read_csv("C://Users//DELL//Desktop//sentiment//sentiment.csv",encoding="latin1",names=column_name)
print(df)
print(df[["exp","comments"]])
df['com'] = df['comments']
print(df["com"])
import re
pr=r'[^\w\s]'
df["coms"]=df["com"].replace(pr," ",regex=True)
print(df["coms"])
import re
pe=r'http'
df["coms"]=df["coms"].replace(pe," ",regex=True)
print(df["coms"])
import re
pe1 = "[0-9]"
df["coms"] =df["coms"].replace(pe1," ",regex=True)
print(df["coms"])
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

df['coms'] = df['coms'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
print(df["coms"])

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
df['coms'] = df.apply(lambda row: nltk.word_tokenize(row['coms']), axis=1)
print(df['coms'])
from nltk.stem import PorterStemmer, WordNetLemmatizer
porter_stemmer = PorterStemmer()
df['coms'] = df['coms'].apply(lambda x: [porter_stemmer.stem(y) for y in x])
print(df["coms"])
df["coms"]=df["coms"].astype(str)
print(df["coms"].dtype)
X= df.iloc[:,-1]
Y= df.iloc[:,0:1]
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer

 #Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

#  Fit and Transform
tfidf_matrix = tfidf_vectorizer.fit_transform(X_train)
tfid_X_Test=tfidf_vectorizer.transform(X_test)

 # Check shape of TF-IDF matrix
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

# tfidf_matrix
Y
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(tfidf_matrix, Y_train)
predict=lr.predict(tfid_X_Test)
print(predict)
from sklearn.metrics import confusion_matrix,accuracy_score
c_result=confusion_matrix(predict,Y_test)
acc_score=accuracy_score(predict,Y_test)
print(c_result)
print(acc_score)

# ex=["he is bad person"]
# ex_count=tfidf_vectorizer.transform(ex)
# print(ex_count)


# predicts=lr.predict(ex_count)
# print(predicts)
# # if predicts==[0]:
# #     print("it is sad statement")
# # else:
# #     print("it is positive statement")    

import pickle

# Save the trained logistic regression model as a pickle file
with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(lr, file)

# Save the TF-IDF vectorizer as a pickle file
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)
