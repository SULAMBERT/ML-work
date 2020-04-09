from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import  pandas as pd
from scipy.sparse import coo_matrix, hstack
import pickle

class mLmodel(object):

    tfidf = None

    def tf_idf(self):
        if self.tfidf is None:
            self.tfidf = TfidfVectorizer(stop_words = 'english', ngram_range=(1,2))
            # , max_features = 50000
        return self.tfidf

    def fit_tfidf(self, data):
        tfidf = self.tf_idf()
        tfidf.fit(data['case_description'])
        X = tfidf.transform(data['case_description'])
        return X

    def split_data(self,data):
        X = data.case_description
        y = data.type
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
        return X_train, X_test, y_train, y_test

    def fit_pipeline(self, data):
        rf = RandomForestClassifier(class_weight="balanced")
        cv = CountVectorizer(max_features=40000, ngram_range=(1, 3))
        pipeline = Pipeline([
            ('feats', FeatureUnion([
                    ('vectorizer', cv),
                    ('ave', AverageWordLengthExtractor())
                                ])
            ),
            ('classifier', rf)
        ])
        X_train, X_test, y_train, y_test = self.split_data(data)
        X_train = np.array(list(X_train))
        sentiment_fit = pipeline.fit(X_train, y_train)
        y_pred = sentiment_fit.predict(X_test)
        print(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print("accuracy score: {0:.2f}%".format(accuracy * 100))
        print(classification_report(y_test, y_pred))

    def fit_model(self,data):
        rf = RandomForestClassifier(class_weight="balanced",n_jobs=-1)
        tfidf = self.tf_idf()
        tfidf.fit(data['case_description'])
        # saving the tokenizer .sav format
        tokenizer_file_sav = 'tokenizer_sv_all.sav'
        pickle.dump(tfidf,open(tokenizer_file_sav,'wb'))
        #saving the tokenizer .pkl format
        tokenizer_file_pickle = 'tokenizer_pk_all.pkl'
        with open(tokenizer_file_pickle, 'wb') as file_:
            pickle.dump(tfidf,file_)
        case_description = tfidf.transform(data['case_description'])
        y = data.type
        data = data.drop('type',axis=1)
        data = data.drop('case_description', axis=1)
        data = data.drop('user_specialty', axis=1)
        A = coo_matrix(case_description)
        B = coo_matrix(data.values)
        #try
        X= hstack([A, B],format='csr')
        print(type(X))
        # X = X.toarray()
        # print("done")
      
        #try.......

        # X = hstack([A, B]).toarray()
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
        print("will start training")
        rf.fit(X_train,y_train)
        #saving the .sav model in order to load for prediction
        model_sv = 'model_sv_all.sav'
        pickle.dump(rf,open(model_sv,'wb'))
        #saving the .pkl model in order to load for prediction
        model_pickle = 'model_pk_all.pkl'
        with open(model_pickle,'wb') as file_m:
            pickle.dump(rf,file_m)
        y_pred = rf.predict(X_test)
        # print(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print("accuracy score: {0:.2f}%".format(accuracy * 100))
        print(classification_report(y_test, y_pred))

class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def average_word_length(self, name):
        """Helper code to compute average word length of a name"""
        return np.mean([len(word) for word in name.split()])

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        df = pd.Series(df)
        return df.apply(self.average_word_length)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self

class ArrayCaster(BaseEstimator, TransformerMixin):
  def fit(self, x, y=None):
    return self

  def transform(self, data):
    return np.transpose(np.matrix(data))
