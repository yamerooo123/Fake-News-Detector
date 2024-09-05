# Fake News Detector

**Deep Learning Machine University Project!**

**OVERVIEW**

We have decided to make an improvement on this research project: [Fake News Detector RNN](https://www.kaggle.com/code/muhammadwaseem123/fake-news-detector-rnn).

We trained our model with the RNN and LSTM algorithm with "secretto hush-hush" optimization to improve the model's accuracy by translating data to different languages. Feel free to use our model!

**GENERAL**

We basically applied back translation to overcome overfitting problem during model training. We also reduce the learning rate to improve the accuracy of our model. The back translation converts English to Japanese to Spanish then convert it back to English. This allows the model to be more complex resulting in "IT BECOMES SMARTER".

**RESEARCHES**

Back Translation - https://www.mdpi.com/2076-3417/13/24/13207

Back Translation Implementation - https://www.fiive.se/en/blog/backtranslation-for-ner

Googletrans - https://pypi.org/project/googletrans/#description

Data Preprocessing - https://www.javatpoint.com/data-preprocessing-machine-learning

**DATASETS**

Training dataset(from kaggle) - https://drive.google.com/file/d/1gD_Q-ksCZlJfgKA22qxpt3TujxLBw4JT/view?usp=sharing

Testing dataset(from hugging_face) - https://drive.google.com/file/d/1fOVo2Wh4scjYNs7PjEA-wHtc3wlWGQfe/view?usp=share_link

Back translated trained dataset - https://drive.google.com/drive/folders/194XyOld3xy0rNo-5QGjM_DSsuC0zbNDI

![image](https://github.com/user-attachments/assets/72aade97-bc93-49d7-a43a-80d3af4a433c)

**(Figure 1. Cleaned dataset)**

**STEPS**

1. Download the dataset from [Kaggle](https://www.kaggle.com/code/muhammadwaseem123/fake-news-detector-rnn) This data will be used as the training model.

2. Download the data from [HuggingFace](https://drive.google.com/file/d/1fOVo2Wh4scjYNs7PjEA-wHtc3wlWGQfe/view). This data will be used as the testing model.

3. Clean training & testing dataset(including applying the back translation) using this [1stCode](https://github.com/yamerooo123/Political-Fake-News-Detector-NLP/blob/main/0.PreprocessedDataset.ipynb) and [2ndCode](https://github.com/yamerooo123/Political-Fake-News-Detector-NLP/blob/main/4.Back_Translate.ipynb)

![image](https://github.com/user-attachments/assets/2d405136-bc4e-4dd9-9f89-f688f8b18179)**(Figure 2. Back translated dataset)**


5. Use LSTM baseline model as stated in the Overview section.

6. TBC...
