import sys

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import defaultdict

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from tqdm import tqdm
import csv

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
MAX_LENGTH = 160
BATCH_SIZE = 16
EPOCHS = 2

# NOTE: You can find commented explanations and visualizations in the bert.ipynb file
# This is just the python script for training the model on the training set and obtaining predictions from the test set
def get_body_dict(data_dir):
    with open(data_dir, encoding='utf_8') as tb:
        body_text = list(csv.reader(tb))
        body_text_dict = {}
        for i, line in enumerate(tqdm(body_text)):
            if i > 0:
                id = int(line[0])
                body_text_dict[id] = line[1]
    return body_text_dict


def get_article_data(data_dir, body_text_dict):
    print("Getting article data")
    with open(data_dir, encoding='utf_8') as ts:
        stances_text = list(csv.reader(ts))

        headlines, bodies, stances = [], [], []

        for i, line in enumerate(tqdm(stances_text)):
            if i > 0:
                body_id = int(line[1].strip())

                stances.append(line[2].strip())
                headlines.append(line[0].strip())
                bodies.append(body_text_dict[body_id])
        return stances, headlines, bodies


# Create a class that initializes the header body text, stances, tokenizer and max length
# The class creates an encoding for the data using the parameters selected and returns the encoding
class FNCDataSet(Dataset):

    def __init__(self, hb_text, stances, tokenizer, max_len):
        self.hb_text = hb_text
        self.stances = stances
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.hb_text)

    def __getitem__(self, item):
        hb_text = str(self.hb_text[item])
        stances = self.stances[item]

        # Create encoding
        encoding = self.tokenizer.encode_plus(
            hb_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'hb_text': hb_text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'stances': torch.tensor(stances, dtype=torch.long)
        }


# Method to use the FNCDataSet class and get the encodings wrapped in a Data Loader
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = FNCDataSet(
        hb_text=df['hb'].to_numpy(),
        stances=df['s_int'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=2
    )


# Create a classifier class that instiates the BERT model, tuned dropout rates for regularization and obtain a fully
# connected layer for the output as well as cross entropy
class NewsClassifier(nn.Module):

    def __init__(self, n_classes):
        super(NewsClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


# Method used for training each epoch
# Uses dataloader and saves values to GPU and takes outputs and uses the
# argmax function to get predictions
def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        stances = d["stances"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Get predictions and loss
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, stances)

        correct_predictions += torch.sum(preds == stances)
        losses.append(loss.item())

        loss.backward()
        # clip gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


# Helper method to evalute model using the data loaders
# Returm accuracy by comparing the predictions to the correct stances/targets
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            stances = d["stances"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, stances)

            correct_predictions += torch.sum(preds == stances)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def train_data(data_dir):
    print("Reading in training data")
    train_stances_path = data_dir + "/train_stances.csv"
    train_body_path = data_dir + "/train_bodies.csv"

    train_bodies_dict = get_body_dict(train_body_path)
    s, h, b = get_article_data(train_stances_path, train_bodies_dict)

    data_df = pd.DataFrame(list(zip(h, b, s)), columns=['h', 'b', 's'])

    data_df['hb'] = data_df['h'] + '[SEP]' + data_df['b']
    data_df['s_int'] = data_df.apply(lambda row: stance_map.get(row['s']), axis=1)

    print("Splitting data")
    # Get the training set and validation set using train_test_split
    df_train, df_val = train_test_split(data_df, test_size=0.1, random_state=RANDOM_SEED)

    print("Creating tokenizer")
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    print("Creating training and validation data loaders")
    # Instatiate training data loader and validation data loader
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LENGTH, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LENGTH, BATCH_SIZE)

    # Gather data using the training data loader and check sizes if the input ids, attention mask and stances (targets)
    data = next(iter(train_data_loader))
    print("Keys in training data loader to model: {}".format(data.keys()))

    print("Inspecting shape...")
    print("Input ids size: {}".format(data['input_ids'].shape))
    print("Attention mask size: {}".format(data['attention_mask'].shape))
    print("Stances (targets) size: {}".format(data['stances'].shape))

    print("Instantiating news classifier")
    # Create an instance of the classifier and use the GPU
    model = NewsClassifier(len(stance_map))
    model = model.to(device)

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)

    print("Softmax Propbability Distributions: {}".format(F.softmax(model(input_ids, attention_mask), dim=1)))

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    # TRAINING BERT MODEL
    # Start the training processes using the BERT model and AdamW optimizer from Hugging Face
    # using some of the recommended parameters for fine tuning:
    # Batch size: 16, 32
    # Learning rate (Adam): 5e-5, 3e-5, 2e-5
    # Number of epochs: 2, 3, 4

    print("Training the bert model")
    # Run model by taking the history and best accuracy for all epochs and using helper methods
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(EPOCHS):

        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(df_val)
        )

        print(f'Val loss {val_loss} accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state2.bin')
            best_accuracy = val_acc

    # Gather training accuracies and validation accuracies from training set
    history['train_acc'][0].item()

    train_acc_list = []
    for x in history['train_acc']:
        train_acc_list.append(x.item())

    val_acc_list = []
    for x in history['val_acc']:
        val_acc_list.append(x.item())

    print(f'Training Accuracies: {train_acc_list}')
    print(f'Validation Accuracies: {val_acc_list}')

    return model, tokenizer, loss_fn


# Helper function to get predictions from model and test data loader
# This function also uses the softmax function for getting probability distributions
def get_predictions(model, data_loader):
    model = model.eval()

    hb_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["hb_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            stances = d["stances"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            hb_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(stances)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return hb_texts, predictions, prediction_probs, real_values


def test_data(data_dir, model, tokenizer, loss_fn):
    print("Reading in test data")
    # Read in test data (headline, body, stances/targets)
    test_stances_path = data_dir + "/competition_test_stances.csv"
    test_body_path = data_dir + "/competition_test_bodies.csv"

    test_bodies_dict = get_body_dict(test_body_path)
    stest, htest, btest = get_article_data(test_stances_path, test_bodies_dict)

    test_data_df = pd.DataFrame(list(zip(htest, btest, stest)), columns=['h', 'b', 's'])

    test_data_df['hb'] = test_data_df['h'] + '[SEP]' + test_data_df['b']
    test_data_df['s_int'] = test_data_df.apply(lambda row: stance_map.get(row['s']), axis=1)

    # Creating test data loader
    print("Creating test data loader")
    test_data_loader = create_data_loader(test_data_df, tokenizer, MAX_LENGTH, BATCH_SIZE)

    # Determine accuracy of model on the test data
    test_acc, _ = eval_model(
        model,
        test_data_loader,
        loss_fn,
        device,
        len(test_data_df)
    )

    print(f'Model on test accuracy: {test_acc.item()}')

    print("Getting predictions for test data using model")
    # Use the helper function to get probabilities on the test data and return the headline body text
    y_hb_texts, y_pred, y_pred_probs, y_test = get_predictions(
        model,
        test_data_loader
    )

    report = classification_report(y_test, y_pred, target_names=stance_map.keys())

    print(f'Classification Report: {report}')

    return y_hb_texts, y_pred, y_pred_probs, y_test


if __name__ == '__main__':
    stance_map = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
    stance_map_inv = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}

    data_dir = sys.argv[1]
    model, tokenizer, loss_fn = train_data(data_dir)
    y_hb_texts, y_pred, y_pred_probs, y_test = test_data(data_dir, model, tokenizer, loss_fn)

    # Saving answer to csv
    answer_df = pd.read_csv(data_dir + "/competition_test_stances_unlabeled.csv")
    answer_df['Stance'] = pd.DataFrame(y_pred.numpy()).apply(lambda row: stance_map_inv.get(row[0]), axis=1)

    answer_df.to_csv(f'{data_dir}/answer.csv', index=False, encoding='utf-8')
