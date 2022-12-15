from acppred.models import Model
from sklearn.ensemble import RandomForestClassifier
import os

def test_model_creation ():
    model = Model(
        estimator=RandomForestClassifier(),
        positive_peptides='data/raw/positive.txt',
        negative_peptides='data/raw/negative.txt'
    )

def test_model_train():
    model = Model(
        estimator=RandomForestClassifier(),
        positive_peptides='data/raw/positive.txt',
        negative_peptides='data/raw/negative.txt'
    )
    model.train()

def test_model_train_return_report():
    model = Model(
        estimator=RandomForestClassifier(),
        positive_peptides='data/raw/positive.txt',
        negative_peptides='data/raw/negative.txt'
    )
    report = model.train()
    assert isinstance(report, str)

def test_model_transform():
    model = Model(
        estimator=RandomForestClassifier(),
        positive_peptides='data/raw/positive.txt',
        negative_peptides='data/raw/negative.txt'
    )
    peptide = 'AAAG'
    X_transform = model.transform([peptide])
    assert X_transform.iloc[0]['A'] == 0.75
    assert X_transform.iloc[0]['G'] == 0.25

def test_model_transform_illegal_aminoacids():
    model = Model(
        estimator=RandomForestClassifier(),
        positive_peptides='data/raw/positive.txt',
        negative_peptides='data/raw/negative.txt'
    )
    peptide = 'AAAGO!'
    X_transform = model.transform([peptide])
    assert X_transform.iloc[0]['A'] == 0.75
    assert X_transform.iloc[0]['G'] == 0.25

def test_models_save():
    model = Model(
        estimator=RandomForestClassifier(),
        positive_peptides='data/raw/positive.txt',
        negative_peptides='data/raw/negative.txt'
    )
    
    filename = 'data/tests/model.pickle'
    # conferindo se o arquivo existe na pasta

    if os.path.isfile(filename):
        os.remove(filename) # remover a versão anterior
    
    model.save('data/tests/model.pickle')
    assert os.path.isfile(filename)



