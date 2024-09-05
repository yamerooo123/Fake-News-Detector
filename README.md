# Fake News Detector - Deep Learning Machine University Project

We have decided to make an improvement on this research project: [Fake News Detector RNN](https://www.kaggle.com/code/muhammadwaseem123/fake-news-detector-rnn).
and use this as the baseline model 

## Base Model
We use dataset from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification), split it train:test 80:20
, use model architect of this notebook [Fake News Detector RNN](https://www.kaggle.com/code/muhammadwaseem123/fake-news-detector-rnn). Create a ModelCheckpoint callback that saves the model's weights every 5 epochs ,Create an EarlyStopping callback that stops training when the validation loss has not improved for 3 consecutive epochs here is result:

![baseline](model/EscoreBaseModel.png)

![baseline](model/basemodel.png)

**Model Accuracy:**
The training accuracy is very high (almost 100%) after just a few epochs, while the validation accuracy is much lower and remains almost flat, around 96%.

The gap between training accuracy and validation accuracy is significant, which could be an indicator of overfitting.

**Model Loss:**
The training loss is very low, near zero, while the validation loss starts at a moderate value and keeps increasing.

The growing gap between training and validation loss is another indicator of overfitting. This shows that while the model is learning the training data perfectly, it is struggling to generalize to the validation data.

## Model 1
We do the same but we use dataset from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) for trainning only and use another source [HuggingFace](https://huggingface.co/datasets/Cartinoe5930/Politifact_fake_news)
for test, here is the result:

![baseline](model/EscoreModel1.png)

![baseline](model/model1.png)

**Model Accuracy:**
Training accuracy quickly reaches almost 100%, while validation accuracy starts to plateau after a few epochs and fluctuates slightly.

**Model Loss:**
Training loss decreases steadily, but validation loss increases after the first few epochs, showing an increasing divergence between training and validation loss.

## Model 2

We apply data augmentation techniques (synonym replacement, random insertion, and random deletion) to increase training data diversity, and by increasing the embedding size and LSTM units for richer word representations and better pattern recognition. Regularization techniques like increased dropout and L2 regularization were added to prevent overfitting. The batch size was reduced for better generalization, and early stopping and checkpoints were implemented to optimize training. These changes enhance the model's ability to generalize, handle complex patterns, and reduce overfitting, leading to better fake news detection performance.

![baseline](model2/EscoreModel2.png)

![baseline](model2/model2result.png)

**Model Accuracy:** The training accuracy reaches near 100%, but the validation accuracy is relatively stable, around 98.5%, with smaller fluctuations compared to the first image.

**Model Loss:** The training loss decreases sharply, and while the validation loss fluctuates, it remains relatively stable and lower than in the first set of plots.
The second model shows less fluctuation in the validation accuracy and smaller divergence between the training and validation losses, even though there's some fluctuation in validation loss.

The model2 is better overall because:
 - It has more stable validation accuracy with fewer fluctuations.
 - The validation loss does not increase drastically like in the model 1.


## **RESEARCHES & REFERENCE**

Back Translation - https://www.mdpi.com/2076-3417/13/24/13207

Back Translation Implementation - https://www.fiive.se/en/blog/backtranslation-for-ner

Googletrans - https://pypi.org/project/googletrans/#description

Data Preprocessing - https://www.javatpoint.com/data-preprocessing-machine-learning

Basic NLP Augmentation - https://maelfabien.github.io/machinelearning/NLP_8/#
