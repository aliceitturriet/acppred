from acppred.models import Model
from argparse import ArgumentParser
from Bio import SeqIO
import pandas as pd

def main():

    argument_parser = ArgumentParser(description="Predicts anticancer peptides in a FASTA file")
    argument_parser.add_argument('--input', required=True, help='Input FASTA file')
    argument_parser.add_argument('--output', required=True, help='Output CSV file')
    argument_parser.add_argument('--model', required=True, help='Pre-trained ACPPred model')

    arguments = argument_parser.parse_args()

    model = Model.load(arguments.model)

    predictions = []

    for sequence_record in SeqIO.parse(arguments.input, 'fasta'):
        sequence = str(sequence_record.seq)
        prediction = model.predict(sequence)
        sequence_data = {'sequence_id': sequence_record.id, 'prediction': prediction}
        predictions.append(sequence_data)

    df_predictions = pd.DataFrame(predictions, columns=['sequence_id', 'prediction'])
    df_predictions.to_csv(arguments.output, index=False)

if __name__ == '__main__':
    main()