# conda install pandas
# conda install -c pytorch pytorch torchvision torchaudio cudatoolkit=10.2
# conda install -c fastchan transformers


# Imports
import math
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer


# Choose device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Device selected: " + device)


# Define data reading options
positive_data_path = "positive_merged.csv.zip"
negative_data_path = "negative_merged_typeCorrect.csv.zip"
bert_model_path = "dbmdz/distilbert-base-turkish-cased"

data_types = {
    "tweetid": "str",
    "userid": "str",
    "user_display_name": "str",
    "user_screen_name": "str",
    "user_reported_location": "str",
    "user_profile_description": "str",
    "follower_count": "int",
    "following_count": "int",
    "account_creation_date": "str",
    "account_language": "str",
    "tweet_language": "str",
    "tweet_text": "str",
    "tweet_time": "str",
    "tweet_client_name": "str",
    "is_retweet": "bool",
    "like_count": "int",
    "retweet_count": "int",
    "hashtags": "str",
    "urls": "str",
    "user_mentions": "str",
}

print("Loading data")
positive_raw_data = pd.read_csv(positive_data_path, dtype=data_types, keep_default_na=False, index_col=False)
positive_raw_data["label"] = 1.0
# print(positive_raw_data.dtypes)

negative_raw_data = pd.read_csv(negative_data_path, dtype=data_types, keep_default_na=False, index_col=False)
negative_raw_data["label"] = 0.0
# print(negative_raw_data.dtypes)
print("Data loaded")
print("Label ratio: {} positive".format(
    positive_raw_data.shape[0] / (positive_raw_data.shape[0] + negative_raw_data.shape[0])
))

negative_raw_data[["account_creation_date", "account_creation_time"]] = negative_raw_data.account_creation_date.str.split(" ", 1, expand=True)
negative_raw_data = negative_raw_data.drop(["account_creation_time"], axis = 1)
# print(negative_raw_data["account_creation_date"])

merged_data = pd.concat([positive_raw_data, negative_raw_data], ignore_index=True)
merged_data = merged_data.sample(frac=1)


# Add extra columns
today = pd.Timestamp("2021-12-01")
merged_data["account_creation_date"] = pd.to_datetime(merged_data.account_creation_date)
merged_data["account_age"] = (today - merged_data.account_creation_date).dt.days
merged_data["is_turkish_tweet"] = merged_data.tweet_language == "tr"
merged_data["is_turkish_account"] = merged_data.account_language == "tr"
merged_data["is_android"] = merged_data.tweet_client_name == "Twitter for Android"
merged_data["is_iphone"] = merged_data.tweet_client_name == "Twitter for iPhone"
merged_data["is_mobile"] = merged_data.is_android & merged_data.is_iphone
merged_data["contains_hashtag"] = len(merged_data.hashtags) > 0
merged_data["contains_url"] = len(merged_data.urls) > 0
print("Finished deriving columns")


# Split sets
def split_data(data, train=0.98, test=0.01, eval=0.01):
    assert train + test + eval == 1.0

    num_examples = data.shape[0]
    data = data.sample(frac=1)

    train_data = data[:int(num_examples * train)]
    test_data = data[int(num_examples * train):int(num_examples * (train+test))]
    eval_data = data[int(num_examples * (train+test)):]

    return (train_data, test_data, eval_data)

train_data, test_data, eval_data = split_data(merged_data)
print("Data split")

# Save data
train_data_path = "train_data.zip"
test_data_path = "test_data.zip"
eval_data_path = "eval_data.zip"
print("Examples per split: {} / {} / {} (train / test / eval)".format(
    train_data.shape[0],
    test_data.shape[0],
    eval_data.shape[0],
))

train_data.to_csv(train_data_path, index=False)
test_data.to_csv(test_data_path, index=False)
eval_data.to_csv(eval_data_path, index=False)
print("Data saved")


# Define features
float_features_names = [
    "follower_count",
    "following_count",
    "like_count",
    "retweet_count",
    "account_age",
    "is_retweet",  # binary
    "is_mobile",  # binary
    "is_turkish_tweet",  # binary
    "is_turkish_account",  # binary
    "contains_hashtag",  # binary
    "contains_url",  # binary
]


# Define model
tokenizer = AutoTokenizer.from_pretrained(bert_model_path)

def process_data(batch):
    feature_dict = tokenizer(
        batch["tweet_text"].tolist(),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    numerical_data = batch[float_features_names].astype(float)
    feature_dict["float_features"] = torch.from_numpy(numerical_data.values).float()

    labels = batch["label"].astype(float)
    feature_dict["labels"] = torch.from_numpy(labels.values).float()

    return feature_dict


class BertTweetClassifier(nn.Module):
    def __init__(self, bert_size, bert_red_size, numeric_feature_size, red_size, freeze_bert=True, dropout_rate=0.1):
        super().__init__()
        self.bert_size = bert_size

        # Use pre-trained BERT model
        self.bert = AutoModel.from_pretrained(
            bert_model_path,
            output_hidden_states=True,
            output_attentions=True,
        )

        for param in self.bert.parameters():
            if freeze_bert:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.weights = nn.Parameter(torch.rand(7, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(bert_size, bert_red_size)
        self.fc2 = nn.Linear(bert_red_size + numeric_feature_size, red_size)
        self.fc3 = nn.Linear(red_size, 1)  # binary classfication
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_dict):
        input_ids = feature_dict["input_ids"]
        float_features = feature_dict["float_features"]

        all_hidden_states, all_attentions = self.bert(
            input_ids,
            output_attentions=True,
            output_hidden_states=True,
        )[-2:]
        batch_size = input_ids.shape[0]

        ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(7, batch_size, 1, self.bert_size)
        atten = torch.sum(ht_cls * self.weights.view(7, 1, 1, 1), dim=[1, 3])

        atten = F.softmax(atten.view(-1), dim=0)
        feature = torch.sum(ht_cls * atten.view(7, 1, 1, 1), dim=[0, 2])

        bert_red_out = self.fc1(self.dropout(feature))
        bert_red_activ_out = self.relu(bert_red_out)

        concat_layer = torch.cat((bert_red_activ_out, float_features), 1)
        concat_out = self.fc2(concat_layer)
        concat_activ_out = self.relu(concat_out)

        out = self.fc3(concat_activ_out)
        prediction = self.sigmoid(out)

        return prediction


# Define evaluation metrics
def report_eval_metrics(model, test_data, step=0.01, batch_size=256):
    print("Reporting metrics")
    counts = {}
    metrics = {}
    num_examples = test_data.shape[0]

    if device == "cuda":
        torch.cuda.empty_cache()
    model.zero_grad()

    for threshold in np.arange(0.0, 1.0 + step, step):
        counts[threshold] = np.zeros(4)  # TP, TN, FP, FN in that order
        metrics[threshold] = np.zeros(5)  # accuracy, precision, recall, specificity, f1_score

    with torch.no_grad():
        for batch_start_index in range(0, num_examples, batch_size):
            batch_num = int(batch_start_index / batch_size)
            # if batch_num % 25 == 0:
            #     print(f"Batch {batch_num} started")

            batch_data = test_data[batch_num : min(batch_num + batch_size, num_examples)]
            feature_dict = process_data(batch_data).to(device)
            predictions = np.squeeze(model(feature_dict))
            labels = feature_dict["labels"]

            for threshold in np.arange(0.0, 1.0 + step, step):
                binary_predictions = (predictions > threshold).float()

                counts[threshold][0] += torch.bitwise_and(binary_predictions == 1.0, labels == 1.0).sum()
                counts[threshold][1] += torch.bitwise_and(binary_predictions == 0.0, labels == 0.0).sum()
                counts[threshold][2] += torch.bitwise_and(binary_predictions == 1.0, labels == 0.0).sum()
                counts[threshold][3] += torch.bitwise_and(binary_predictions == 0.0, labels == 1.0).sum()

            del predictions

            if device == "cuda":
                torch.cuda.empty_cache()

    for threshold in np.arange(0.0, 1.0 + step, step):
        metrics[threshold][0] = ((counts[threshold][0] + counts[threshold][1]) / num_examples).item()
        metrics[threshold][1] = (counts[threshold][0] / (counts[threshold][0] + counts[threshold][2])).item()
        metrics[threshold][2] = (counts[threshold][0] / (counts[threshold][0] + counts[threshold][3])).item()
        metrics[threshold][3] = (counts[threshold][1] / (counts[threshold][2] + counts[threshold][1])).item()
        metrics[threshold][4] = ((2*counts[threshold][0]) / (2*counts[threshold][0] + counts[threshold][2] + counts[threshold][3])).item()

    return metrics


def train_model(train_data, test_data, batch_size=32, learning_rate=1e-5, num_epochs=3, freeze_bert=True, dropout_rate=0.1):
    # Initialize model
    model = BertTweetClassifier(768, 64, len(float_features_names), 32, freeze_bert=freeze_bert, dropout_rate=dropout_rate).to(device)
    print("Model created")

    pre_training_metrics = report_eval_metrics(model, test_data, step=0.01)
    pre_training_metrics_path = "pointers/pre_training_metrics_bs{}_lr{}_fb{}_dp{}_ep{}.pkl".format(
        batch_size,
        learning_rate,
        0 if freeze_bert else 1,
        dropout_rate,
        num_epochs,
    )

    with open(pre_training_metrics_path, "wb") as f:
        pickle.dump(pre_training_metrics, f)


    # Launch training
    print("Training started")
    print("Parameters: batch size = {}, learning rate = {}, epochs = {}, freeze bert = {}, dropout rate = {}".format(
        batch_size,
        learning_rate,
        num_epochs,
        freeze_bert,
        dropout_rate,
    ))
    num_examples = train_data.shape[0]
    criterion = nn.BCELoss()

    if device == "cuda":
        torch.cuda.empty_cache()
    model.zero_grad()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    batch_count = math.ceil(num_examples / batch_size)

    for epoch_num in range(num_epochs):
        print(f"Epoch {epoch_num} started")
        train_data = train_data.sample(frac=1)  # Shuffle the batches each epoch
        total_epoch_loss = 0

        for batch_start_index in range(0, num_examples, batch_size):
            model.zero_grad()

            batch_num = int(batch_start_index / batch_size)
            if batch_num % 250 == 0:
                print(f"Train batch {batch_num} started")

            batch_data = train_data[batch_num : min(batch_num + batch_size, num_examples)]

            feature_dict = process_data(batch_data).to(device)
            predictions = np.squeeze(model(feature_dict))
            labels = feature_dict["labels"]

            loss = criterion(predictions, labels)
            total_epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            del predictions
            if device == "cuda":
                torch.cuda.empty_cache()

        print(f"Epoch {epoch_num} finished")
        print("Average batch loss: " + str(total_epoch_loss / batch_count))

        intra_training_metrics = report_eval_metrics(model, test_data, step=0.01)
        intra_training_metrics_path = "pointers/intra_training_metrics_bs{}_lr{}_fb{}_dp{}_ep{}_{}.pkl".format(
            batch_size,
            learning_rate,
            0 if freeze_bert else 1,
            dropout_rate,
            epoch_num + 1,
            num_epochs,
        )

        with open(intra_training_metrics_path, "wb") as f:
            pickle.dump(intra_training_metrics, f)

    # Save and evaluate trained model
    print("Training done")
    print("Saving model data")
    model_path = "pointers/model_bs{}_lr{}_ep{}.pt".format(batch_size, learning_rate, num_epochs)
    torch.save(model.state_dict(), model_path)

    post_training_metrics = report_eval_metrics(model, test_data, step=0.01)
    post_training_metrics_path = "pointers/post_training_metrics_bs{}_lr{}_fb{}_dp{}_ep{}.pkl".format(
        batch_size,
        learning_rate,
        0 if freeze_bert else 1,
        dropout_rate,
        num_epochs,
    )

    with open(post_training_metrics_path, "wb") as f:
        pickle.dump(post_training_metrics, f)


# Define hyperparamter sweep values
batch_size_vals = [32, 64]
learning_rate_vals = [1e-5, 1e-3, 1e-1]
num_epochs_vals = [3, 5, 10]
freeze_bert_vals = [True, False]
dropout_rate_vals = [0.1, 0.5, 0.75]

for batch_size in batch_size_vals:
    for learning_rate in learning_rate_vals:
        for num_epochs in num_epochs_vals:
            for freeze_bert in freeze_bert_vals:
                for dropout_rate in dropout_rate_vals:
                    # Launch each training instance
                    train_model(
                        train_data,
                        test_data,
                        batch_size,
                        learning_rate,
                        num_epochs,
                        freeze_bert,
                        dropout_rate,
                    )
