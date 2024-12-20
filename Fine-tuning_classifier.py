import os
import pandas as pd
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)



class PeptideDataset(Dataset):
    r"""PyTorch Dataset class for loading data.
    This is where the data parsing happens.
    This class is built with reusability in mind: it can be used as is as.
    Arguments:
      path (:obj:`str`):
          Path to the data partition.
    """
    def __init__(self, data):
        self.texts = []
        self.labels = []
        for seq in data['Sequence']:
            self.texts.append(seq)
        for lab in data['label']:
            self.labels.append(lab)

        self.n_examples = len(self.labels)

        return

    def __len__(self):

        return self.n_examples

    def __getitem__(self, item):

        return {'text': self.texts[item],
                'label': self.labels[item]}




class Gpt2ClassificationCollator(object):
    r"""
    Data Collator used for GPT2 in a classificaiton rask.
    It uses a given tokenizer and label encoder to convert any text and labels to numbers that
    can go straight into a GPT2 model.
    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed
    straight into the model - `model(**batch)`.
    Arguments:

      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.

      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to
          labels names and Values map to number associated to those labels.

      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.

    """
    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):
        self.use_tokenizer = use_tokenizer
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        self.labels_encoder = labels_encoder

        return

    def __call__(self, sequences):
        r"""
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this
        class as a function.

        Arguments:

          item (:obj:`list`):
              List of texts and labels.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """

        texts = [sequence['text'] for sequence in sequences]
        labels = [sequence['label'] for sequence in sequences]
        # Encode all labels using label encoder.
        labels = [self.labels_encoder[label] for label in labels]
        # Call tokenizer on all texts to convert into tensors of numbers with
        # appropriate padding.
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,
                                    max_length=self.max_sequence_len)
        # Update the inputs with the associated encoded labels as tensor.
        inputs.update({'labels': torch.tensor(labels)})
        return inputs


def train(model, dataloader, optimizer, scheduler, device):
    r"""
    Train pytorch model on a single pass through the data loader.
    It will use the global variable `model` which is the transformer model
    loaded on `_device` that we want to train on.
    This function is built with reusability in mind: it can be used as is as long
      as the `dataloader` outputs a batch in dictionary format that can be passed
      straight into the model - `model(**batch)`.
    Arguments:

        dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
            Parsed data into batches of tensors.

        optimizer_ (:obj:`transformers.optimization.AdamW`):
            Optimizer used for training.

        scheduler_ (:obj:`torch.optim.lr_scheduler.LambdaLR`):
            PyTorch scheduler.

        device (:obj:`torch.device`):
            Device used to load tensors before feeding to model.

    Returns:

        :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
          Labels, Train Average Loss].
    """

    # Use global variable for model.
    #global model

    # Tracking variables.
    predictions_labels = []
    true_labels = []
    # Total loss for this epoch.
    total_loss = 0

    # Put the model into training mode.
    model.train()

    # For each batch of training data...
    for batch in tqdm(dataloader, total=len(dataloader)):
        # Add original labels - use later for evaluation.
        true_labels += batch['labels'].numpy().flatten().tolist()

        # move batch to device
        batch = {k: v.type(torch.long).to(device) for k, v in batch.items()}


        model.zero_grad()
        outputs = model(**batch)

        # The call to `model` always returns a tuple, so we need to pull the
        # loss value out of the tuple along with the logits. We will use logits
        # later to calculate training accuracy.
        loss, logits = outputs[:2]
        total_loss += loss.item()
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        # Convert these logits to list of predicted labels values.
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()

    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)

    # Return all true labels and prediction for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss


def validation(model, dataloader, device):
    r"""Validation function to evaluate model performance on a
    separate set of data.

    This function will return the true and predicted labels so we can use later
    to evaluate the model's performance.

    This function is built with reusability in mind: it can be used as is as long
      as the `dataloader` outputs a batch in dictionary format that can be passed
      straight into the model - `model(**batch)`.

    Arguments:

      dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
            Parsed data into batches of tensors.

      device (:obj:`torch.device`):
            Device used to load tensors before feeding to model.

    Returns:

      :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
          Labels, Train Average Loss]
    """

    # Use global variable for model.
    #global model

    predictions_labels = []
    predictions_probs = []
    true_labels = []
    total_loss = 0
    model.eval()

    # Evaluate data for one epoch
    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch['labels'].numpy().flatten().tolist()

        batch = {k: v.type(torch.long).to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
            loss, logits = outputs[:2]
            #logits = logits.detach().cpu().numpy()
            probs = F.softmax(logits, dim=1)

            total_loss += loss.item()
            predict_content = logits.argmax(axis=-1).flatten().tolist()
            predictions_labels += predict_content
            predictions_probs.extend(probs[:,1].tolist())
    avg_epoch_loss = total_loss / len(dataloader)

    return true_labels, predictions_labels, avg_epoch_loss, predictions_probs




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='gpu or not')
    parser.add_argument('--random_seed', default=666, type=str, required=False, help='random seed')
    parser.add_argument('--raw_data_path', default='/data/train_dat.csv', type=str, required=False, help='trainset')
    parser.add_argument('--batch_size', default=32, type=int, required=False, help='batch size')
    parser.add_argument('--epochs', default=8, type=int, required=False, help='Training loop')
    parser.add_argument('--lr', default=1.5e-3, type=float, required=False, help='learning rate')
    parser.add_argument('--max_length', default=50, type=int, required=False, help='max_length')
    parser.add_argument('--warmup_steps', default=25000, type=int, required=False, help='warm up steps')  
    parser.add_argument('--model_path', default='/AMP_models/ProteoGPT/', type=str, required=False, help='pretrained model path')
    parser.add_argument('--output_path', default='/Classifier/best_model.pt', type=str, required=False, help='model output path')
    parser.add_argument('--trainset_path', default='/Classifier/trainset.csv', type=str,required=False, help='trainset path')
    parser.add_argument('--validset_path', default='/Classifier/validset.csv', type=str, required=False, help='validset path')
    parser.add_argument('--testset_path', default='/Classifier/testset.csv', type=str, required=False, help='testset path')
    parser.add_argument('--loss_path', default='/Classifier/loss.png', type=str, required=False, help='loss figure path')
    parser.add_argument('--acc_path', default='/Classifier/accuracy.png', type=str, required=False, help='accuracy figure path')
    parser.add_argument('--metrix_path', default='/Classifier/confusion_matrix.png', type=str, required=False, help='confusion_matrix figure path')
    parser.add_argument('--predict_path', default='/Classifier/predicted_result.csv', type=str, required=False, help='predicted_result path')
    parser.add_argument('--roc_path', default='/Classifier/roc.png', type=str, required=False, help='predicted_result path')
    parser.add_argument('--report_path', default='/Classifier/report.txt', type=str, required=False, help='evaluation_report path')
    parser.add_argument('--allloss_path', default='/Classifier/all_loss.csv', type=str, required=False, help='all_loss path')
    parser.add_argument('--allacc_path', default='/Classifier/all_acc.csv', type=str, required=False, help='all_acc path')
    parser.add_argument('--auc_path', default='/Classifier/auc.csv', type=str, required=False, help='auc path')




    args = parser.parse_args()
    print('args:\n' + args.__repr__())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)
    random_seed = args.random_seed
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    max_length = args.max_length
    warmup_steps = args.warmup_steps
    raw_data_path = args.raw_data_path
    model_path = args.model_path
    output_path = args.output_path
    loss_path = args.loss_path
    acc_path = args.acc_path
    metrix_path = args.metrix_path
    predict_path = args.predict_path
    roc_path = args.roc_path
    report_path = args.report_path
    allloss_path = args.allloss_path
    allacc_path = args.allacc_path
    auc_path = args.auc_path
    trainset_path = args.trainset_path
    validset_path = args.validset_path
    testset_path = args.testset_path

    labels_ids = {'neg': 0, 'pos': 1}
    n_labels = len(labels_ids)

    print('Loading tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # default to left padding
    tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token

    # Get model configuration.
    #print('Loading configuraiton...')
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_path, num_labels=n_labels)

    # Get the actual model.
    print('Loading model...')
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path, config=model_config)

    # resize model embedding to match new tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id

    # Load model to defined device.
    model.to(device)
    print('Model loaded to `%s`' % device)
    print(model)
    # Create data collator to encode text and labels into numbers.
    gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer,
                                                              labels_encoder=labels_ids,
                                                              max_sequence_len=max_length)

    data = pd.read_csv(raw_data_path)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=random_seed)
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=random_seed)

    print('Dealing with Train...')
    train_dataset = PeptideDataset(data=train_data)
    print('Created `train_dataset` with %d examples!' % len(train_dataset))

    # 将序列保存到 CSV 文件
    sequences = train_dataset.texts
    train_dataset_file = pd.DataFrame(sequences, columns=['Sequence'])
    train_dataset_file.to_csv(trainset_path, index=False)
    print(f"Sequences have been saved to {trainset_path}")


    # Move pytorch dataset into dataloader.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=gpt2_classificaiton_collator)
    print('Created `train_dataloader` with %d batches!' % len(train_dataloader))

    print('Dealing with Validation...')
    # Create pytorch dataset.
    valid_dataset = PeptideDataset(data=val_data)
    print('Created `valid_dataset` with %d examples!' % len(valid_dataset))

    # 将序列保存到 CSV 文件
    sequences = valid_dataset.texts
    valid_dataset_file = pd.DataFrame(sequences, columns=['Sequence'])
    valid_dataset_file.to_csv(validset_path, index=False)
    print(f"Sequences have been saved to {validset_path}")

    # Move pytorch dataset into dataloader.
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=gpt2_classificaiton_collator)
    print('Created `eval_dataloader` with %d batches!' % len(valid_dataloader))

    print('Dealing with Test...')
    # Create pytorch dataset.
    test_dataset = PeptideDataset(data=test_data)
    print('Created `test_dataset` with %d examples!' % len(test_dataset))

    # 将序列保存到 CSV 文件
    sequences = test_dataset.texts
    test_dataset_file = pd.DataFrame(sequences, columns=['Sequence'])
    test_dataset_file.to_csv(testset_path,index=False)
    print(f"Sequences have been saved to {testset_path}")

    # Move pytorch dataset into dataloader.
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                  collate_fn=gpt2_classificaiton_collator)
    print('Created `test_dataloader` with %d batches!' % len(test_dataloader))

    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=True)
    total_steps = len(train_dataloader) * epochs
    print(total_steps)
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    # Store the average loss after each epoch so we can plot them.
    all_loss = {'train_loss': [], 'val_loss': []}
    all_acc = {'train_acc': [], 'val_acc': []}
    best_auc = 0
    # Loop through each epoch.
    print('Epoch')
    for epoch in tqdm(range(epochs)):
        print('Training on batches...')
        # Perform one full pass over the training set.
        train_labels, train_predict, train_loss = train(model, train_dataloader, optimizer, scheduler, device)
        train_acc = accuracy_score(train_labels, train_predict)

        # Get prediction form model on validation data.
        print('Validation on batches...')
        valid_labels, valid_predict, val_loss, val_probs = validation(model, valid_dataloader, device)
        val_acc = accuracy_score(valid_labels, valid_predict)

        fpr, tpr, _ = roc_curve(valid_labels, valid_predict)
        val_auc = auc(fpr, tpr)

        # Print loss and accuracy values to see how training evolves.
        print("train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f" % (train_loss, val_loss, train_acc, val_acc))
        print()

        # Store the loss value for plotting the learning curve.
        all_loss['train_loss'].append(train_loss)
        all_loss['val_loss'].append(val_loss)
        all_acc['train_acc'].append(train_acc)
        all_acc['val_acc'].append(val_acc)

        #select the best model
        if val_auc > best_auc:
            torch.save(model.state_dict(), output_path)
            best_auc = val_auc

    loss_df = pd.DataFrame(all_loss)
    acc_df = pd.DataFrame(all_acc)
    loss_df.to_csv(allloss_path, index=None)
    acc_df.to_csv(allacc_path, index=None)

    ###model evaluation
    # Plot loss curves.
    plot_dict(all_loss,
              use_xlabel='Epochs',
              use_ylabel='Loss Value',
              use_linestyles=['-', '--'],
              width=2,
              height=1,
              use_dpi=600,
              use_title='Loss Curves of the Prediction of Testset',
              path= loss_path)


    # Plot accuracy curves.
    plot_dict(all_acc,
              use_xlabel='Epochs',
              use_ylabel='Accuracy Value',
              use_linestyles=['-', '--'],
              width=2,
              height=1,
              use_dpi=600,
              use_title='Accuracy Curves of the Prediction of Testset',
              path= acc_path)



    #load the best model
    model.load_state_dict(torch.load(output_path))

    # Get prediction form model on validation data. This is where you should use your test data.
    true_labels, predictions_labels, avg_epoch_loss, predictions_probs = validation(model, test_dataloader, device)

    peptide = test_dataloader.dataset[:]['text']


    df = pd.DataFrame({
        'Peptide' : peptide,
        'True Labels' : true_labels,
        'Predicted Labels' : predictions_labels,
        'Predicted Probabilities' : predictions_probs

    })

    df.to_csv(predict_path)

    #plot the roc curve
    fpr, tpr, _ = roc_curve(true_labels, predictions_probs)
    auc_df = pd.DataFrame({'FPR': fpr,
                           'TPR': tpr})

    auc_df.to_csv(auc_path, index=None)
    #print(fpr)
    #print('=' * 50)
    #print(tpr)
    #print('=' * 50)
    roc_auc = auc(fpr, tpr)
    #print(roc_auc)
    #print('=' * 50)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(roc_path, dpi=600)
    plt.show()


    # Create the evaluation report.
    evaluation_report = classification_report(true_labels, predictions_labels, labels=list(labels_ids.values()), target_names=list(labels_ids.keys()))
    # Show the evaluation report.
    print(evaluation_report)
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(evaluation_report)
        f.close()


    # Plot confusion matrix.
    plot_confusion_matrix(y_true=true_labels,
                          y_pred=predictions_labels,
                          classes=list(labels_ids.keys()),
                          normalize=True,
                          width=1,
                          height=1,
                          use_dpi=600,
                          path=metrix_path)




if __name__ == '__main__':
    main()
