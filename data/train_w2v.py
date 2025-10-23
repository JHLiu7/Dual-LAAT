# %%
import numpy as np
import pandas as pd

# %%
import gensim.models.word2vec as w2v

data_dir = 'coding_data'
split_dir = 'splits'
output_dir = 'w2v_v2'

# %%
model_configs = {
    "vector_size": 100,
    "min_count": 0,
    "workers": 4,
    "epochs": 5,
}
min_document_count = 3

# %%
# use all train docs to train word2vec model
def get_train_docs(version):
    if version == 'iii':
        data = pd.read_feather(f'{data_dir}/mimiciii_icd9.feather')
        splt = pd.read_feather(f'{split_dir}/mimiciii_clean_splits.feather')
    else:
        data = pd.read_feather(f'{data_dir}/mimiciv_icd{version}.feather')
        splt = pd.read_feather(f'{split_dir}/mimiciv_icd{version}_split.feather')

    print("all rows", len(data))

    train_ids = splt[splt['split'] == 'train']['_id'].values
    train_data = data[data['_id'].isin(train_ids)]
    train_docs = train_data['text'].tolist()
    print("train rows", len(train_docs))
    
    return train_docs

# %%
doc_iii = get_train_docs('iii')
doc9 = get_train_docs('9')
doc10 = get_train_docs('10')



# %%
def word_tokenizer(string): 
    return string.split()


# %%
# docs = doc9[:100] + doc10[:100]
docs = doc9 + doc10 + doc_iii
docs_token = [word_tokenizer(doc) for doc in docs]

# %%
model = w2v.Word2Vec(docs_token, **model_configs)

# %%
# remove rare words
word_counts = dict()
for sentence in docs:
    words_in_sentence = set()
    for word in sentence.split():
        if word not in word_counts:
            word_counts[word] = 0
        if word not in words_in_sentence:
            word_counts[word] += 1
        words_in_sentence.add(word)

words_to_remove = [
    word
    for word, count in word_counts.items()
    if (count < min_document_count) and (word in model.wv.key_to_index)
]
ids_to_remove = [model.wv.key_to_index[word] for word in words_to_remove]


# %%
for word in words_to_remove:
    del model.wv.key_to_index[word]

model.wv.vectors = np.delete(model.wv.vectors, ids_to_remove, axis=0)

for i in sorted(ids_to_remove, reverse=True):
    del model.wv.index_to_key[i]

for i, key in enumerate(model.wv.index_to_key):
    model.wv.key_to_index[key] = i
    
    
print(model.wv.vectors.shape)

# add <UNK> and <PAD> tokens
vec = np.random.randn(model.vector_size)
model.wv.add_vector('<UNK>', vec)
model.init_sims(replace=True) # normalize weights
model.wv.add_vector('<PAD>', np.zeros(model.vector_size))



# %%
# save model
model.save(f"{output_dir}/word2vec.model")

import pickle

t2i = {t:i for i, t in enumerate(model.wv.index_to_key)}

with open(f'{output_dir}/token2id.pkl', 'wb') as f:
    pickle.dump(t2i, f)

np.save(f'{output_dir}/vectors.npy', model.wv.vectors)

# %%
# load model
model = w2v.Word2Vec.load(f"{output_dir}/word2vec.model")

# %%
print(model.wv.vectors.shape)

# %%



