import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


sample_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
sample_contents = []
for file in sample_files:
    with open(file, 'r', encoding='utf-8') as f:
        sample_contents.append(f.read())

vectorize = lambda Text: TfidfVectorizer().fit_transform(Text).toarray()
similarity = lambda doc1, doc2: cosine_similarity([doc1, doc2])

vectors = vectorize(sample_contents)
s_vectors = list(zip(sample_files, vectors))

