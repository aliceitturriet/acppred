from Bio.SeqUtils import ProtParam
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from acppred.utils import ALLOWED_AMINOACIDS
import pandas as pd
import pickle

class Model: 
    def __init__(self, estimator, positive_peptides, negative_peptides):
        """
        This class defines an estimator for anticancer peptide prediction

        Args:
            - estimator: a scikit-learn estimator
            - positive_peptides: a file containing anticancer peptides
            - negative_peptides: a file containing non-anticancer peptides
        """
        self.estimator = estimator
        self.positive_peptides = positive_peptides
        self.negative_peptides = negative_peptides

    def transform(self, X):
        """
        Transform a set of protein sequences into aminoacid percents

        Args:
            - X: a list of protein sequences
        """
        X_transform = []

        for peptide in X: 
            peptide = ''.join([aminoacid for aminoacid in peptide.upper() if aminoacid in ALLOWED_AMINOACIDS])
            aa_percent = ProtParam.ProteinAnalysis(peptide).get_amino_acids_percent()
            X_transform.append(aa_percent)

        return pd.DataFrame(X_transform)

    def train(self):
        """
        Trains a predictive model for anticancer peptides
        """
        X = []
        y = []

        with open(self.positive_peptides) as reader:
            for peptide in reader:
                X.append(peptide)
                y.append(1)
        
        with open(self.negative_peptides) as reader:
            for peptide in reader:
                X.append(peptide)
                y.append(0)

        X_transform = self.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_transform, y)
        self.estimator.fit(X_train, y_train)
        y_pred = self.estimator.predict(X_test)
        report = classification_report(y_test, y_pred)
        return report

    def predict(self, sequence):
        """
        Predicts the anticancer activity for a given peptide

        Args:
            - sequence: a peptide sequence to be analyzed

        """
        X_transform = self.transform([sequence])
        return self.estimator.predict_proba(X_transform)[0][1]

    def save(self, filename):
        """
        Saves the model to a file

        Args:
            - filename: path to the trained model file
        """
        with open(filename, 'wb') as writer:
            writer.write(pickle.dumps(self))

    @staticmethod
    def load(filename):
        """
        Loads a trained model objects

        Args:
            - filename: path to the trained model file
        """
        with open(filename, 'rb') as reader:
            return pickle.loads(reader.read())