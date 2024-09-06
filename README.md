# Fake News Detector - Deep Learning University Project

- 6311374 Suphawith Phusanbai
- 6420063 Kritsada Kruapat
  
We have decided to make an improvement on this [Fake News Detector RNN](https://www.kaggle.com/code/muhammadwaseem123/fake-news-detector-rnn) notebook and use this as the baseline model (lstm model)
## Clean & tokenization data :

[Clean & tokenize Kaggle dataset](https://drive.google.com/file/d/1gD_Q-ksCZlJfgKA22qxpt3TujxLBw4JT/view?usp=share_link)

[Clean & tokenize Hugging Face dataset](https://drive.google.com/file/d/1fOVo2Wh4scjYNs7PjEA-wHtc3wlWGQfe/view?usp=sharing)

1. **Remove Unnamed Columns:** Any unnamed columns are removed to clean up the dataset.
2. **Handle Missing Values:** Missing or empty values in the `back_translated_text` column are filled with the word "missing."
3. **Convert to Lowercase:** All text in the `back_translated_text` column is converted to lowercase.
4. **Remove Punctuation:** Punctuation is removed from the text to leave only words and spaces.
5. **Remove Stopwords:** Common words (like "the", "and", "is") that don't add much meaning are removed using NLTK’s stopwords.
6. **Tokenization:** The cleaned text is split into individual words (tokens), excluding non-alphanumeric characters.
7. **Lemmatization:** Words are reduced to their base form (e.g., "running" becomes "run") to standardize them for better processing.

# .
#  🔎🔬🧑🏽‍🔬🧪🔬🔎 Time to Experiment 🔎🧑🏽‍🔬🔬🧪🧪🔬
## Base Model
We use dataset from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification), Clean & tokenization data  then split it train:test 80:20
, use model architect of this notebook [Fake News Detector RNN](https://www.kaggle.com/code/muhammadwaseem123/fake-news-detector-rnn).
Create a ModelCheckpoint callback that saves the model's weights every 5 epochs ,Create an EarlyStopping callback that stops training when the validation loss has not improved for 3 consecutive epochs here is result:

![img](02_ModelBase/EscoreBaseModel.png)

![img](02_ModelBase/basemodel.png)

**Model Accuracy:**
The training accuracy is very high (almost 100%) after just a few epochs, while the validation accuracy is much lower and remains almost flat, around 96%.

The gap between training accuracy and validation accuracy is significant, which could be an indicator of overfitting.

**Model Loss:**
The training loss is very low, near zero, while the validation loss starts at a moderate value and keeps increasing.

The growing gap between training and validation loss is another indicator of overfitting. This shows that while the model is learning the training data perfectly, it is struggling to generalize to the validation data.

## Model 1
We do the same but we use dataset from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) for trainning only and use another source [HuggingFace](https://huggingface.co/datasets/Cartinoe5930/Politifact_fake_news)
for test, here is the result:

![img](03_Model1/EscoreModel1.png)

![img](03_Model1/model1.png)

**Model Accuracy:**
Training accuracy quickly reaches almost 100%, while validation accuracy starts to plateau after a few epochs and fluctuates slightly.

**Model Loss:**
Training loss decreases steadily, but validation loss increases after the first few epochs, showing an increasing divergence between training and validation loss.

## Model 2

We apply data augmentation techniques (synonym replacement, random insertion, and random deletion) to increase training data diversity, and by increasing the embedding size and LSTM units for richer word representations and better pattern recognition. Regularization techniques like increased dropout and L2 regularization were added to prevent overfitting. The batch size was reduced for better generalization, and early stopping and checkpoints were implemented to optimize training. These changes enhance the model's ability to generalize, handle complex patterns, and reduce overfitting, leading to better fake news detection performance.

![img](04_Model2/EscoreModel2.png)

![img](04_Model2/model2result.png)

**Model Accuracy:** The training accuracy reaches near 100%, but the validation accuracy is relatively stable, around 98.5%, with smaller fluctuations compared to the first image.

**Model Loss:** The training loss decreases sharply, and while the validation loss fluctuates, it remains relatively stable and lower than in the first set of plots.
The second model shows less fluctuation in the validation accuracy and smaller divergence between the training and validation losses, even though there's some fluctuation in validation loss.

## Model 3

we use google api translate for doing back translation from English -> japanese -> Spanish -> English, aim to reduce the overfitting if this project,
this model use dataset from back translation algorithm then clean and do tokenization data (29995/72134 , we use 2-3 days to get this data because of the limited time of this project so do back translation only the first 30K which having Length of tokenized text column: 8654262)

this model use back translation dataset only for train and use hugging_face for test 
here is the result:

![img](05_Model3/EscoreModel3.png)
![img](05_Model3/model3.png)

the model performs well on the training data, the increasing validation loss and fluctuating accuracy indicate overfitting.

## Model 4
We use combine original dataset(kaggle) + back-translate + basic augmention (synonym replacement, random insertion, and random deletion) as train dataset, Length of combined dataset (original + augmented): 408516 

and do the same as model4
here is the result:

![img](06_Model4/EscoreModel4.png)
![img](06_Model4/model4Result.png)

**Accuracy:** The training accuracy reaches almost 100%, but there is some fluctuation in validation accuracy, which indicates some overfitting.

**Loss:** The validation loss increases as the epochs go by, while the training loss keeps decreasing, which is a sign of overfitting.

as this time model 2 is Best balance between training and validation performance


# Application:
This is for who want to quickly assess the authenticity of news articles and understand the likelihood of encountering fake news based on content analysis. 

How It Works:
1. **Enter URL:** Users can input the URL of a news article they want to analyze. The app scrapes the text from the webpage and analyzes it for authenticity.

2. **Preprocess the Text:** The app processes the text using a pre-trained tokenizer to convert it into sequences that the machine learning model can understand.

3. **Model Prediction:** The app runs the preprocessed text through a machine learning model to determine whether the article is likely fake. It displays the result as a percentage chance of being fake.

![img](streamlit.png)

we will push it on hugghing space which you can play or try our model soon! let's finish model part first

## **RESEARCHES & REFERENCE**

Back Translation - https://www.mdpi.com/2076-3417/13/24/13207

Back Translation Implementation - https://www.fiive.se/en/blog/backtranslation-for-ner

Googletrans - https://pypi.org/project/googletrans/#description

Data Preprocessing - https://www.javatpoint.com/data-preprocessing-machine-learning

Basic NLP Augmentation 

https://maelfabien.github.io/machinelearning/NLP_8/#

https://www.kaggle.com/code/andreshg/nlp-glove-bert-tf-idf-lstm-explained/notebook

