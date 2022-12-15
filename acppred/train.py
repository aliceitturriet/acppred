from sklearn.ensemble import RandomForestClassifier
from acppred.models import Model
from argparse import ArgumentParser

def main():

    argument_parser = ArgumentParser(description='Trains an anticancer peptide classification model')
    argument_parser.add_argument(
        '--positive-peptides', 
        default='data/raw/positive.txt',
        help='a file containing anticancer peptides'
    )
    argument_parser.add_argument(
        '--negative-peptides',
        default='data/raw/negative.txt',
        help='a file containing non-anticancer peptides'
    )
    argument_parser.add_argument(
        '--output',
        help='path to the output trained model',
        required=True
    )
    argument_parser.add_argument(
        '--show-report',
        help='shows the classification report after training',
        action='store_true'
    )

    arguments = argument_parser.parse_args()

    model = Model(
        estimator=RandomForestClassifier(), 
        positive_peptides=arguments.positive_peptides, 
        negative_peptides=arguments.negative_peptides
    )

    report = model.train()

    if arguments.show_report:
        print(report)

    model.save(arguments.output)

if __name__ == '__main__':
    main()

