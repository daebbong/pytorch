# make the one-hot vector based sklearn

from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

corpus = ['Time flies like an arrow.',
          'Fruit flies like a banana.']

one_hot_vectorizer = CountVectorizer(binary=True)
print(one_hot_vectorizer.fit_transform(corpus))
one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()
print(one_hot)
vocab = one_hot_vectorizer.get_feature_names_out()
sns.heatmap(one_hot,annot=True,cbar=False,
            xticklabels=vocab,yticklabels=['Sentence 1','Sentence 2'])
plt.show()
