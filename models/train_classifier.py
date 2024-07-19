"""
Disaster Response Pipeline Project

Contains code to: load and preprocess text, train model, evaluate model, and save model.
"""
import sys

import pandas as pd
import pickle
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    """
    Loads data from database.

    :param database_filepath: str path to database file
    :return: Feature matrix, label matrix and labels
    :rtype: tuple
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_response_cleaned', engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])

    return X, Y, Y.columns


def tokenize(text):
    """
    Tokenizes and lemmatizes text.

    :param text: str text to tokenize
    :return: tokenized text
    :rtype: list
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Builds and returns model pipeline object.

    :return: model pipeline object
    :rtype: Pipeline
    """
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('mlpc', MultiOutputClassifier(clf))
    ])

    parameters = {
        # 'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'mlpc__estimator__n_estimators': [50, 100, 200],
        # 'mlpc__n_jobs': -1
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates model performance.

    :param model: model pipeline object
    :param X_test: test feature matrix
    :param Y_test: test label matrix
    :param category_names: label names
    :return: None
    :rtype: None
    """
    y_pred = model.predict(X_test)
    y_test_arr = Y_test.to_numpy()

    for i in range(y_test_arr.shape[1]):
        print(f"\nClassification Report for Output {category_names[i]}:")
        print(classification_report(y_test_arr[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Saves model to file.

    :param model:
    :param model_filepath: str path to save model file
    :rtype: None
    :return: None
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
