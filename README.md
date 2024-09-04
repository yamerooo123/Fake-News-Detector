# Fake News Detector

**Deep Learning Machine University Project!**

We have decided to make an improvement on this research project: [Fake News Detector RNN](https://www.kaggle.com/code/muhammadwaseem123/fake-news-detector-rnn).

We trained our model with the RNN algorithm with "secretto hush-hush" optimization to improve the model's accuracy. Feel free to use our model!

**Overview Concept**

We basically applied back translation to overcome overfitting problem during model training. We also reduce the learning rate to improve the accuracy of our model. The back translation converts English to Japanese and Spanish then convert it back to English. This allows the model to be more complex which resulting in "THEY BECOME SMARTER".

**Researches:**

Back Translation - https://www.mdpi.com/2076-3417/13/24/13207

Back Translation Implementation - https://www.fiive.se/en/blog/backtranslation-for-ner

Googletrans - https://pypi.org/project/googletrans/#description

Data Preprocessing - https://www.javatpoint.com/data-preprocessing-machine-learning

**Kaggle Dataset:**
[Fake News Classification Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification?resource=download&select=WELFake_Dataset.csv)

Training dataset(from kaggle) - https://drive.google.com/file/d/1gD_Q-ksCZlJfgKA22qxpt3TujxLBw4JT/view?usp=sharing

Testing dataset(from hugging_face) - https://drive.google.com/file/d/1fOVo2Wh4scjYNs7PjEA-wHtc3wlWGQfe/view?usp=share_link

![image](https://github.com/user-attachments/assets/48ea163d-617b-47e4-afbf-4cceb86363ae)

**Steps**

1. Download the dataset from [Kaggle](https://www.kaggle.com/code/muhammadwaseem123/fake-news-detector-rnn) This data will be used as a training model.

2. Download the data from [HuggingFace](https://drive.google.com/file/d/1fOVo2Wh4scjYNs7PjEA-wHtc3wlWGQfe/view). This data will be used as a testing model.

3. Clean training & testing dataset(including applying the back translation)

4. Use LSTM baseline model as stated in the Overview section.

5. TBC...
