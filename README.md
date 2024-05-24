# Sentiment Analysis using Gated Recurrent Units
by Utsav A. Negi

## Aim
The aim of this report is to demonstrate the approach used to train neural networks built using Gated
Recurrent Unit (GRU) for sentiment analysis and to evaluate its performance in predicting the sentiments
of a given sentence.

## Source Code Details
For this assignment, the details about the source code are as follows:
<ul>
  <li>hw9.py: Source code consisting of all the essential classes, functions & main ML pipeline for
training and evaluating a GRU-based neural network which is trained and tested using the given
data.csv file.</li>
    <li>hw9_extra_200.py: Source code consisting of all the essential classes, functions and main ML
pipeline for training and evaluating a GRU-based neural network which is trained and tested using
SentimentDataset200.</li>
  <li>hw9_extra_400.py: Source code consisting of all the essential classes, functions and main ML
pipeline for training and evaluating a GRU-based neural network which is trained and tested using
SentimentDataset400.</li>
</ul>

## Data Organization
For this project, the data from each dataset is organized into training data and testing data by using a list
of sentences alongside a list of their associated sentiments. The TextDataset class which is a child class
of torch.utils.data.Dataset is responsible to generate embeddings and subsequent sentiments in one-hot
encoding form. To generate the embeddings, every subwords in the sentences are tokenized and later
embedded using pretrained Bert tokenizer and pretrained Bert model respectively. The __getitem__ class
function is overridden to return a pair consisting of embedding and its associated sentiments in one-hot
encoding form.

## Model
The model uses nn.GRU from PyTorch to keep track of hidden layers and one of the constructor
parameters of the model class is responsible for setting bidirectional to either True or False. Furthermore,
at the start of the model, the hidden layer is initialized as a zero vector which is then fed into the GRU
layer along with the input. Non-linearity function is applied to the resultant tensor and using the linear
layer and LogSoftmax function, the output of desired size is achieved. Note that, in this case, the input
data will be arranged in batches, which gives more freedom to the GRU to handle the values in the hidden
states.

## Training
The dataloader yields a pair consisting of processed embedding and its associated sentiments in one-hot
encoding form. The embedding is reshaped to the dimensions (batch size, max length of embedding,
input size) which is later serves as an input for the GRU-based model. The output of the model is then
compared with the ground truth sentiment by determining loss using Negative Log Likelihood loss
function. The parameters are optimized using Adam optimizer with a learning rate of 1e-3. For the
training, the batch size is set to 25, input size is set to 768, the epochs are set to 25, embedding size is set
to 512, hidden layer size is set to 800 along with number of layers set to 2. Furthermore, the output layer
is set to 3 in the case of using data.csv as the data source while the output layer is set to 2 in the case of
using SentimentDataset200 and SentimentDataset400.

## Evaluation
The trained model is evaluated by determining its accuracy in determining correct sentiments for test
sentences and the observations are then presented using a confusion matrix. The dataloader yields a pair
consisting of processed embedding and its associated sentiments in one-hot encoding form. The
embedding is again reshaped to match the dimension format used in training which is used as an input to
the trained model. The batched generated results and the batched ground truth sentiments are iterated
to determine the classification accuracy and the values for the confusion matrix are recorded.
