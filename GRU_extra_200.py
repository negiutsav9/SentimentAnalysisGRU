'''################################################################################################
Filename: hw9_extra_200.py
Author: Utsav Negi
Purpose: Source code consisting of all the essential classes, functions and main ML pipeline for
    training and evaluating GRU-based neural network which is trained and tested using
    SentimentDataset200.
################################################################################################'''

import gc
import os
import gzip
import torch
import random
import pickle
import numpy as np
import seaborn as sn
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.optim import Adam
from transformers import DistilBertModel, DistilBertTokenizer
from torch.utils.data import DataLoader, Dataset

from google.colab import drive
drive.mount('/content/drive')

# setting seeds for consistency (reference: Lecture 2 and HW 2)
seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmarks = False
os.environ["PYTHONHASHSEED"] = str(seed)

# function to create training set and testing set comprising
# of sentences and their associated sentiments
def data_organizer(type):
    if type == 200:
        f = gzip.open("/content/drive/MyDrive/Hw9/data/sentiment_dataset_train_200.tar.gz", 'rb')
        train_dataset = f.read()
        f = gzip.open("/content/drive/MyDrive/Hw9/data/sentiment_dataset_test_200.tar.gz", 'rb')
        test_dataset = f.read()
    else:
        f = gzip.open("/content/drive/MyDrive/Hw9/data/sentiment_dataset_train_400.tar.gz", 'rb')
        train_dataset = f.read()
        f = gzip.open("/content/drive/MyDrive/Hw9/data/sentiment_dataset_test_400.tar.gz", 'rb')
        test_dataset = f.read()
    positive_reviews_train, negative_reviews_train, vocab = pickle.loads(train_dataset, encoding='latin1')
    positive_reviews_test, negative_reviews_test, vocab = pickle.loads(test_dataset, encoding='latin1')
    indexed_dataset_train, indexed_dataset_test = [], []

    for category in positive_reviews_train:
        for review in positive_reviews_train[category]:
            indexed_dataset_train.append([review, category, 1])
    for category in negative_reviews_train:
        for review in negative_reviews_train[category]:
            indexed_dataset_train.append([review, category, 0])
    random.shuffle(indexed_dataset_train)

    for category in positive_reviews_test:
        for review in positive_reviews_test[category]:
            indexed_dataset_test.append([review, category, 1])
    for category in negative_reviews_test:
        for review in negative_reviews_test[category]:
            indexed_dataset_test.append([review, category, 0])
    random.shuffle(indexed_dataset_test)

    return indexed_dataset_train, indexed_dataset_test

# class to convert sentences into embeddings and sentiments into one-hot encoding while returning
# a pair consisting of embedding and its associated one-hot encoded sentiment
class TextDataset(Dataset):
    def __init__(self,dataset):
        super(TextDataset,self).__init__()
        self.sentences = []
        self.sentiments = []
        self.one_hot_sentiments = []
        for data in dataset:
            self.sentences.append(data[0])
            self.sentiments.append(data[2])
        self.word_embeddings = self.create_embeddings()
        for i in range(len(self.sentiments)):
            self.one_hot_sentiments.append(self.sentiment_to_one_hot(self.sentiments[i]))

    # function to convert sentences into embeddings
    def create_embeddings(self):
        distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        distilbert_model = DistilBertModel.from_pretrained('distilbert/distilbert-base-uncased')
        bert_tokenized_sentences_ids = [distilbert_tokenizer.encode(sentence, padding='max_length', truncation=True, max_length=512) for sentence in self.sentences]
        subword_embeddings = []
        for tokens in tqdm(bert_tokenized_sentences_ids):
            input_ids = torch.tensor(tokens).unsqueeze(0)
            with torch.no_grad():
                outputs = distilbert_model(input_ids)
            subword_embeddings.append(outputs.last_hidden_state)
        return subword_embeddings

    # function to convert sentiments into one-hot encodings
    def sentiment_to_one_hot(self, sentiment):
        sentiment_tensor = torch.zeros(2)
        if sentiment == 1:
            sentiment_tensor[1] = 1
        elif sentiment == 0:
            sentiment_tensor[0] = 1
        sentiment_tensor = sentiment_tensor.type(torch.long)
        return sentiment_tensor

    def __len__(self):
        return len(self.word_embeddings)

    def __getitem__(self, index):
        #print(self.word_embeddings[index].shape)
        return self.word_embeddings[index], self.one_hot_sentiments[index]

# function to organize data, create training dataset and testing dataset, and create training
# dataloader and testing dataloader
def createDataLoader(type):
    indexed_dataset_train, indexed_dataset_test = data_organizer(type)
    print('Created Training Set and Testing Set')
    trainDataset = TextDataset(indexed_dataset_train)
    print("Created training dataset")
    testDataset = TextDataset(indexed_dataset_test)
    print("Created testing dataset")
    trainDL = DataLoader(trainDataset, 25, shuffle=True, num_workers=2)
    testDL = DataLoader(testDataset, 25, shuffle=False, num_workers=2)
    print('Created dataloaders')
    return trainDL, testDL

# class to create RNN model
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, bidirectional=False):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=self.bidirectional)
        if(self.bidirectional):
            self.fc = nn.Linear(2*hidden_size, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        if(self.bidirectional):
            init_h = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        else:
            init_h = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        out, _ = self.gru(x, init_h)
        out = self.fc(self.relu(out[:,-1, :]))
        out = self.logsoftmax(out)
        return out

# training function
def training(net, trainDL, net_path, epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    criterion = nn.NLLLoss()
    optimizer = Adam(net.parameters(), lr=1e-3)
    training_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        count = 0
        for i,data in enumerate(trainDL):
            embedding, sentiment = data
            embedding = embedding.reshape(-1, 512, 768)
            embedding = embedding.to(device)
            sentiment = sentiment.to(device)
            optimizer.zero_grad()
            output = net(embedding)
            loss = criterion(output, torch.argmax(sentiment, dim=1))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            count += 1
        avg_loss = running_loss / float(count)
        training_losses.append(avg_loss)
        print("[epoch:%d]   loss: %.5f" % (epoch+1, avg_loss))
    torch.save(net.state_dict(), net_path)
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return training_losses

# function to evaluate the peformance of the model
def validation(net, net_path, testDL):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(net_path))
    net.to(device)
    classification_accuracy = 0.0
    confusion_matrix = torch.zeros(2,2)
    total = 0
    with torch.no_grad():
        for i,data in enumerate(testDL):
            embedding = (data[0].reshape(-1, 512, 768)).to(device)
            sentiment = (data[1]).to(device)
            output = net(embedding)
            predicted_idx = torch.argmax(output.data, 1)
            gt_idx = torch.argmax(sentiment,1)
            for prediction,truth in zip(predicted_idx, gt_idx):
                if prediction == truth:
                    classification_accuracy += 1
                confusion_matrix[truth, prediction] += 1
                total += 1
    print("\nOverall classification accuracy: %0.2f%%" %  (float(classification_accuracy) * 100 /float(total)))
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return confusion_matrix

if __name__ == "__main__":
    
    trainDL_200, testDL_200 = createDataLoader(200)

    # training model with bidirectional=False
    net_200_uni = Model(input_size=768, hidden_size=800, output_size=2, num_layers=2, bidirectional=False)
    uni_training_losses_200 = training(net_200_uni, trainDL_200, "/content/drive/MyDrive/Hw9/net_uni_200.pth",20)
    cmatrix_uni_200 = validation(net_200_uni, "/content/drive/MyDrive/Hw9/net_uni_200.pth", testDL_200)

    # training model with bidirectional=True
    net_200_bi = Model(input_size=768, hidden_size=800, output_size=2, num_layers=2, bidirectional=True)
    bi_training_losses_200 = training(net_200_bi, trainDL_200, "/content/drive/MyDrive/Hw9/net_bi_200.pth",20)
    cmatrix_bi_200 = validation(net_200_bi, "/content/drive/MyDrive/Hw9/net_bi_200.pth", testDL_200)

    plt.figure()
    plt.title("Training Loss of Net when Bidirectional is set to True or False")
    plt.plot(uni_training_losses_200, label="bidirectional = False")
    plt.plot(bi_training_losses_200, label="bidirectional = True")
    plt.legend()
    plt.savefig("/content/drive/MyDrive/Hw9/TrainingLoss_200.png", dpi=400)

    uni_hm = sn.heatmap(
        data = cmatrix_uni_200,
        cmap = 'Blues',
        annot=True,
        fmt='g',
        xticklabels=['negative', 'positive'],
        yticklabels=['negative', 'positive'],
    )
    uni_hm.set(xlabel='Predictions', ylabel='Ground Truth')
    plt.savefig("/content/drive/MyDrive/Hw9/unidirectional_GRU_cmatrix_200.png", dpi=400)

    bi_hm = sn.heatmap(
        data = cmatrix_bi_200,
        cmap = 'Blues',
        fmt='g',
        annot=True,
        xticklabels=['negative', 'positive'],
        yticklabels=['negative', 'positive']
    )
    bi_hm.set(xlabel='Predictions', ylabel='Ground Truth')
    plt.savefig("/content/drive/MyDrive/Hw9/bidirectional_GRU_cmatrix_200.png", dpi=400)