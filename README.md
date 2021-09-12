# Safe-City-Classification
Frontend created using Streamlit library to demostrate the model built for classificaiton of harrasment stories.
Implemented features
1. Get predictions for entered story/description
2. Along with predicted caegory also display the probabilities of the instance belonging to each one of the classes as predicted by the model.

Blog Link : https://ashishsalaskar1.medium.com/tablenet-end-to-end-table-detection-and-tabular-data-extraction-from-scanned-document-images-13c846e8f8f5<br>
Deployment link : https://safe-city-clf.herokuapp.com/

## Introduction
Here our objective is to use short stories which were submitted online and to be able to automatically categorize each story submitted by a user. We will consider the user stories as our training data. We will try to build a Machine Learning model which will take these stories/descriptions as input and try to predict the categories of harassment it belongs to. The main thing to keep in mind here is that a description may belong to multiple categories. For example, a description could indicate both Commenting and Groping cases.<br>
Data-Source : https://github.com/swkarlekar/safecity <br>
Authors: Sweta Karlekar & Mohit Bansal, University of North Carolina at Chapel Hill

The dataset has 7201 training samples, 990 validation samples, and 1701 test samples. Each data sample consists of a Description of the incident followed by whether it belongs to classes Commenting, Staring, and Groping. As it is a Multi-Label classification problem, each data point can belong to multiple classes.

## Performance Metrics
Initially, this problem is a Multi-Label Classification problem, but later we will map it to a Multi-Class classification problem, hence we can use multi-class classification metrics for evaluating our model. We use the following metrics :
LogLoss: It is calculated as the average of negative log probabilities of the data point belonging to the correct class label. Log loss can range from 0 to infinity, lower LogLoss indicates that more points are classified correctly. In our case, LogLoss seems to be very helpful as we have multiple classes.

## Data Preprocessing
We follow the following Sequence to clean the data
- Decontract Words
- Remove Special Symbols
- Remove Stopwords
- Remove HTML tags
- Remove Punctuations
- Lemmatize the sentence using WordNetLemmatizer

## Feature Engineering
We try to extract features from the text descriptions which can help us further in classification. These are the features extracted
- Length of Descriptions & Average Word Length
- Text Scoring Metrics : Difficulty Score, flesch_reading_ease, flesch_kincaid_grade, coleman_liau_index
- Part of Speech Tagging

## TF-IDF Weighted GloVe Embeddings
In order to convert our text descriptions into numerical representations, we use the TF-IDF Weighted GloVe Embeddings techniques using the following steps
- First, we load the pre-trained Glove vector and store the vector embeddings in a dictionary. The glove vector we use provides a 300-dimensional embedding vector for each word.
- For each sentence, we do a weighted TF-IDF sum. We first initialize a vector with 300 zeros. Then for each valid word in the sentence, we add glove_embedding[word] + tf_idf_score[word] to the vector and finally divide it by summation of tf_idf values.

## Model Building
### Custom 2-Step Classifier
As we noticed the majority of the classes belong to class 0, due to this when we observe the Confusion matrix of the trained models we see that the predictions favor Class 0. So in order to fix this issue, we try to build a model which follows a 2-step classification approach.
- Model 1: Classify if the data belongs to class = 0 or not.
For training this model we modify the dataset as follows: if class ≥ 1, the label=0 else label = 1
- Model 2: If model 1 predicts class != 0, then predict the label from class 1–7. For training this model we only use points in the dataset which belong to Class 1–7
- ![](https://miro.medium.com/max/963/1*nR1g36AJvmqFmcmtKWbXqg.png)
Prediction Steps: Here we first pass the data to be predicted to Model 1 first, it checks if the sample belongs to Class 0 or not. If it doesn't belong to class 0 then we pass the sample to Model 2 which can then predict if the sample belongs to Classes 1 to 7.

## Deployment
To provide an intuitive user interface for the users to try out our model, we have deployed the model on Heroku. We provide the feature for the user to enter a short description of the harassment incident. This data entered is then pre-processed and then fed into the ML model which predicts the category to which the incident belongs. Along with the prediction, we also display the probabilities of the incident belonging to each class which our model predicted for further observation of needed. In order quickly to test we also have an option to pick a random sentence from a set of sentences and predict the result just to have a quick look into the working of the app.
Link : https://safe-city-clf.herokuapp.com/

