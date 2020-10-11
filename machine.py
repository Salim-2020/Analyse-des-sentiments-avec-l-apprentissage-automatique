
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
desired_width=320
import numpy as np
pd.set_option('display.width', desired_width)
#_______________________ PART 2___________________________________________________________________________________________________________#

data = pd.read_csv('./avis/nouveau/Carrefour.csv')
print(data)

import re
def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = re.sub('-', ' ', text)  # replace '-' with space
    text = re.sub(r"\d", "", text)  # remove number
    text = re.sub(r"\s+", " ", text, flags=re.I)  # remove space
    text = re.sub(r"[^\w\d'\s]+", '', text)  # remove punctuation sauf '

    return text
data['Texte']        =   data['Texte'].apply(clean_text)

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
tfidf = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7)
features = tfidf.fit_transform(data.Texte).toarray()
labels = data.Label






print("Each of the %d complaints is represented by %d features (TF-IDF score of unigrams and bigrams)" %(features.shape))

X = data['Texte'] # Collection of documents
y = data['Label'] # Target or the labels we want to predict (i.e., the 13 different complaints of products)
X_train, X_test, y_train, y_test = train_test_split(features,
                                                               labels,
                                                                test_size=0.2,
                                                               random_state=1)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
text_classifier = LogisticRegression()
text_classifier.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
predictions = text_classifier.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print("accuracy score test", accuracy_score(y_test, predictions))


