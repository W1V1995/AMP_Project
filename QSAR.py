import numpy as np
import pandas as pd
import argparse
import os
from tqdm import tqdm
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='GPU or not')
    parser.add_argument('--ParM', default=0.9, type=int, required=False, help='para M')
    parser.add_argument('--ParN', default=1.1, type=int, required=False, help='para N')
    parser.add_argument('--nTer', default=1.0, type=int, required=False, help='N-')
    parser.add_argument('--cTer', default=0.0, type=int, required=False, help='C-')
    parser.add_argument('--samples_path', default='Data/Sequence.csv',
                        type=str, required=False, help="raw data path")
    parser.add_argument('--output_path', default='Data/Sequence_RS.csv', type=str,
                        required=False, help='output path')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print('using device:', device)

    ParM = args.ParM
    ParN = args.ParN
    nTer = args.nTer
    cTer = args.cTer
    samples_path = args.samples_path
    output_path = args.output_path


    hScale = {"A": 0.229, "C": 0.274, "D": 0, "E": 0, "F": 0.949, "G": 0, "H": 0.229, "I": 0.873, "K": 0, "L": 0.949,
              "M": 0.631, "N": 0, "P": 0.229, "Q": 0, "R": 0.096, "S": 0, "T": 0.032, "V": 0.599, "W": 1, "Y": 0.484, "X": 0}

    def maxScorePaneJTB2017(L, m=0.9, n=1.1):
        return max([((nTer + cTer + L - Z) ** m) * (((L -Z ) * hScale["R"] + Z) ** n) for Z in range(L)])

    def calcPaneJTB2017(Seq, m=0.9, n=1.1, maxScore=1):
        # Charges from N- and C-terminal groups
        charge = nTer + cTer
        # Count Arginines and Lysines
        charge += (Seq.count("R") + Seq.count("K")) * 1
        # Count Glutamic and Aspartic acids
        charge += (Seq.count("E") + Seq.count("D")) * -1

        # Hydrophobicity
        try:
            H = np.sum([hScale[aa] for aa in Seq])
        except:
            H = 0

        if charge <= 0:
            score = 0
        else:
            score = (charge ** m) * (H ** n) / maxScore

        return score

    """
    ###rs for generator

    df = pd.DataFrame(columns=['Sequence'])
    for file_name in os.listdir(samples_path):
        file_path = os.path.join(samples_path, file_name)
        seqs = pd.read_csv(file_path, header=None, names=['Sequence'])
        df = df.append(seqs, ignore_index=True)
        df = df.drop_duplicates(subset='Sequence')
    """

    ###rs for classifier
    df = pd.read_csv(samples_path)


    #df = pd.read_csv(samples_path, header=None, names=['Sequence'])
    df['Length'] = df['Sequence'].apply(lambda x: len(x))
    df['RS'] = 0
    df['AS'] = 0
    df['maxScore'] = 0

    for i, Seq in enumerate(tqdm(df['Sequence'], desc='Processing')):
        #index = df[df['Sequence'] == Seq].index.tolist()[0]
        maxScore = maxScorePaneJTB2017(len(Seq))
        RS = calcPaneJTB2017(Seq, ParM, ParN, maxScore)
        AS = RS * len(Seq)
        #df['maxScore'][index] = maxScore
        #df['RS'][index] = RS
        #df['AS'][index] = AS

        df['maxScore'][i] = maxScore
        df['RS'][i] = RS
        df['AS'][i] = AS

    df.sort_values(by='RS', inplace=True, ascending=False)
    df.to_csv(output_path, index=None)


if __name__ == '__main__':
    main()