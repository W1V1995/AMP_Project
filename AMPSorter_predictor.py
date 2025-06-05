import os
import pandas as pd
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import (GPT2Config,
                          GPT2Tokenizer,
                          GPT2ForSequenceClassification)


class PeptideDataset(Dataset):

    def __init__(self, data):
        self.texts = []
        for seq in data['Sequence']:
            self.texts.append(seq)

        self.n_examples = len(self.texts)

        return

    def __len__(self):

        return self.n_examples

    def __getitem__(self, item):

        return {'text': self.texts[item]}




class Gpt2ClassificationCollator(object):

    def __init__(self, use_tokenizer, max_sequence_len=None):
        self.use_tokenizer = use_tokenizer
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len

        return

    def __call__(self, sequences):


        texts = [sequence['text'] for sequence in sequences]

        # Call tokenizer on all texts to convert into tensors of numbers with
        # appropriate padding.
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,
                                    max_length=self.max_sequence_len)

        return inputs



def Prediction(model, dataloader, device):


    # Use global variable for model.
    #global model

    predictions_labels = []
    predictions_probs = []


    model.eval()

    # Evaluate data for one epoch
    for batch in tqdm(dataloader, total=len(dataloader)):


        batch = {k: v.type(torch.long).to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs[0]
            probs = F.softmax(logits, dim=1)
            predict_content = logits.argmax(axis=-1).flatten().tolist()
            predictions_labels += predict_content
            predictions_probs.extend(probs[:, 1].tolist())

    return  predictions_labels, predictions_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='gpu or not')
    parser.add_argument('--max_length', default=50, type=int, required=False, help='max_length')
    parser.add_argument('--batch_size', default=32, type=int, required=False, help='batch size')
    parser.add_argument('--raw_data_path', default='/Data/Sequence.csv', type=str, required=False, help='peptide dataset path')
    parser.add_argument('--model_path', default='/AMP_models/ProteoGPT/', type=str, required=False, help='pretrained model path')
    parser.add_argument('--classifier_path', default='/AMP_models/AmpSorter/best_model.pt', type=str, required=False, help='classifier model path')
    parser.add_argument('--output_path', default='/Data/Sequence_pred.csv', type=str, required=False, help='antimicrobial activity prediction results output path')
    parser.add_argument('--candidate_amp_path', default='/Data/Sequence_c_amps.csv', type=str, required=False, help='candidate amp path')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('using device:', device)
    max_length = args.max_length
    batch_size = args.batch_size
    raw_data_path = args.raw_data_path
    model_path = args.model_path
    classifier_path =args.classifier_path
    output_path = args.output_path
    candidate_amp_path = args.candidate_amp_path


    labels_ids = {'neg': 0, 'pos': 1}
    n_labels = len(labels_ids)

    print('Loading tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # default to left padding
    tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token

    # Get model configuration.
    # print('Loading configuraiton...')
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_path, num_labels=n_labels)

    # Get the actual model.
    print('Loading model...')
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path, config=model_config)

    # resize model embedding to match new tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id

    #load th best model
    model.load_state_dict(torch.load(classifier_path, map_location=torch.device('cpu')))

    # Load model to defined device.
    model.to(device)
    print('Model loaded to `%s`' % device)

    # Create data collator to encode text and labels into numbers.
    gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer,
                                                              max_sequence_len=max_length)

    pepbase = pd.read_csv(raw_data_path)


    print('Dealing with Pepbase...')
    predicted_set = PeptideDataset(data=pepbase)
    print('Created `predicted_set` with %d examples!' % len(predicted_set))

    # Move pytorch dataset into dataloader.
    predicted_dataloader = DataLoader(predicted_set, batch_size=batch_size, shuffle=True,
                                  collate_fn=gpt2_classificaiton_collator)
    print('Created `predicted_dataloader` with %d batches!' % len(predicted_dataloader))

    # Get prediction form model on validation data. This is where you should use your test data.
    predictions_labels, predictions_probs = Prediction(model, predicted_dataloader, device)

    peptide = predicted_dataloader.dataset[:]['text']

    df = pd.DataFrame({
        'Sequence': peptide,
        'Predicted Labels': predictions_labels,
        'Predicted Probabilities': predictions_probs
    })

    df_extracted = df[df['Predicted Labels'] == 1]
    df.sort_values(by='Predicted Probabilities', inplace=True, ascending=False)
    df_extracted.sort_values(by='Predicted Probabilities', inplace=True, ascending=False)

    df.to_csv(output_path, index=None)
    df_extracted.to_csv(candidate_amp_path, index=None)


if __name__ == '__main__':
    main()
