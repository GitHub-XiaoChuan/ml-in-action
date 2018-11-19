import nltk
from sklearn.feature_extraction.text import CountVectorizer

sent1 = 'The cat is walking in the bedroom.'
sent2 = 'A dog was running across the kitchen.'

count_vec = CountVectorizer()
sentences = [sent1, sent2]

# [[0 1 1 0 1 1 0 0 2 1 0],[1 0 0 1 0 0 1 1 1 0 1]]
print(count_vec.fit_transform(sentences).toarray())

# ['across', 'bedroom', 'cat', 'dog', 'in', 'is', 'kitchen', 'running', 'the', 'walking', 'was']
print(count_vec.get_feature_names())

# 分词 和 正规化

# ['The', 'cat', 'is', 'walking', 'in', 'the', 'bedroom', '.']
tokens_1 = nltk.word_tokenize(sent1)
print(tokens_1)
# ['A', 'dog', 'was', 'running', 'across', 'the', 'kitchen', '.']
tokens_2 = nltk.word_tokenize(sent2)
print(tokens_2)

# 寻找原始的词根
stemmer = nltk.stem.PorterStemmer()
stem_1 = [stemmer.stem(t) for t in tokens_1]
print(stem_1)
stem_2 = [stemmer.stem(t) for t in tokens_2]
print(stem_2)

# 词性标注
pos_tag_1 = nltk.tag.pos_tag(tokens_1)
print(pos_tag_1)
pos_tag_2 = nltk.tag.pos_tag(tokens_2)
print(pos_tag_2)