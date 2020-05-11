import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer



def preprocess_text(text):
    txt = re.sub('\[.*\]', '', text)  # removes [] type messages
    txt = re.sub('http\S*', '', text)
    txt = re.sub('[0-9]+', '', txt.lower())
    txt = re.sub('\s\s+', ' ', txt)
    txt = re.sub('^\s+', '', txt)
    txt = re.sub('\s$', '', txt)
    txt = txt.rstrip()

    txt = ''.join(txt)
    removed_stop_words = [word for word in txt.split() if word.lower() not in stopwords.words('english')]
    return " ".join(removed_stop_words)

def train():

    pd.set_option('display.max_columns', None)
    df = pd.read_csv('train_dataset.csv')
    df.drop_duplicates(inplace=True)
    df_copy = df['Message_Text'].copy()
    df_copy=df_copy.apply(preprocess_text)

    transformer = CountVectorizer(analyzer=preprocess_text)
    data_transformed=transformer.transform(df_copy)


    X_train,X_test,y_train,y_test = train_test_split(data_transformed,df['label'],test_size=0.20)

    model = LogisticRegression(solver='liblinear',penalty='l1')
    classifier = model.fit(X_train,y_train)
    pred=classifier.predict(X_test)
    print("Test Accuracy Score: ",accuracy_score(y_test,pred))

    with open('vector.pickle','wb') as file:
        pickle.dump(transformer,file)
    with open('classifier.pickle','wb') as file:
        pickle.dump(classifier,file)


def predict():

    df = pd.read_csv('test_dataset.csv')
    with open('vector.pickle','rb') as file:
        transformer = pickle.load(file)

    with open('classifier.pickle', 'rb') as file:
        classifier = pickle.load(file)
    df['Message_Text'] = df['Message_Text'].apply(preprocess_text)

    tr = transformer.transform(df['Message_Text'])
    pred = classifier.predict(tr)
    df['predicted'] = pred
    print("Test Sample Accuracy Score: ",accuracy_score(pred,df['label']))

    df.to_csv('results.csv')
if __name__ == '__main__':
    predict()
