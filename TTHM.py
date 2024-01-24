import pandas as pd
import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
import jieba as jb
import re

file_rcsv = './data/rating4.csv'
file_pcsv = './data/place4.csv'
r_df = pd.read_csv(file_rcsv)
p_df = pd.read_csv(file_pcsv)
r_df['str_userId'] = 99
for i in range(3531):
    mask = r_df['userId']==i
    r_df.loc[mask,'str_userId'] = str(i)
r_df['str_itemId'] = 99
for i in range(266):
    mask = r_df['itemId']==i
    r_df.loc[mask,'str_itemId'] = str(i)
print(r_df.dtypes)
print(r_df.sample(10))
p_df['str_itemId'] = 99
for i in range(266):
    mask = p_df['itemId']==i
    p_df.loc[mask,'str_itemId'] = str(i)
print(p_df.dtypes)
print(p_df.sample(10))

im_path = './data/image/'
for i in range(266):
    mask = p_df['itemId']==i
    p_df.loc[mask,'image'] = im_path + str(i)+'.jpg'
for i in range(266):
    mask = r_df['itemId']==i
    r_df.loc[mask,'image'] = im_path + str(i)+'.jpg'
def visualize(idx):
    current_row = r_df.iloc[idx]
    image = plt.imread(current_row["image"])
    text = current_row["genre"]
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("image")
    plt.show()
    print(text)

random_idx = np.random.choice(len(p_df))
visualize(random_idx)

random_idx = np.random.choice(len(p_df))
visualize(random_idx)


# 定义删除除字母,数字，汉字以外的所有符号的函数
def remove_punctuation(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


# 加载停用词
stopwords = stopwordslist("./data/chineseStopWords.txt")
r_df['clean_genre'] = r_df['genre'].apply(remove_punctuation)
p_df['clean_genre'] = p_df['genre'].apply(remove_punctuation)
r_df.sample()

#分词，并过滤停用词
r_df['cut_genre'] = r_df['clean_genre'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
p_df['cut_genre'] = p_df['clean_genre'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))

r_numeric_feature_names = ['str_userId','title','cut_genre','image']
p_numeric_feature_names = ['title','cut_genre','image']

r_numeric_features = r_df[r_numeric_feature_names]
p_numeric_features = p_df[p_numeric_feature_names]

r_numeric_dataset = tf.data.Dataset.from_tensor_slices(dict(r_numeric_features))
p_numeric_dataset = tf.data.Dataset.from_tensor_slices(dict(p_numeric_features))
p = tf.data.Dataset.from_tensor_slices(p_numeric_features)
print(r_numeric_dataset)
print(p_numeric_dataset)

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range

    return image
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)
ratings_dataset = r_numeric_dataset
places_datasets = p_numeric_dataset
print(len(ratings_dataset))
print(len(places_datasets))
ratings = ratings_dataset.map(lambda x: {
    "title": x["title"],
    "str_user_id": x["str_userId"],
    "cut_genre": x["cut_genre"],
    "image":load_and_preprocess_image(x["image"])
})
places = places_datasets.map(lambda x: {
    "title": x["title"],
    "cut_genre": x["cut_genre"],
    "image":load_and_preprocess_image(x["image"])
})

place_ids = places.batch(1_000).map(lambda x: x["title"])
user_ids = ratings.batch(10000).map(lambda x: x["str_user_id"])
unique_place_ids = np.unique(np.concatenate(list(place_ids)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

user_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_user_ids, mask_token=None),
  # We add an additional embedding to account for unknown tokens.
  tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32)
])

input_shape = (192, 192, 3)
weight_decay = 0.0001
learning_rate = 0.001
label_smoothing = 0.1
validation_split = 0.2
batch_size = 128
num_epochs = 50
patch_size = 2  # Size of the patches to be extracted from the input images.
num_patches = (input_shape[0] // patch_size) ** 2  # Number of patch
embedding_dim = 64  # Number of hidden units.
mlp_dim = 64
dim_coefficient = 4
num_heads = 4
attention_dropout = 0.2
projection_dropout = 0.2
num_transformer_blocks = 8  # Number of repetitions of the transformer layer

print(f"Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ")
print(f"Patches per image: {num_patches}")

#Use data augmentation
# Implement the patch extraction and encoding layer.
class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, self.patch_size, self.patch_size, 1),
            strides=(1, self.patch_size, self.patch_size, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        return tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))


class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        return self.proj(patch) + self.pos_embed(pos)
# Implement the external attention block
def external_attention(
    x, dim, num_heads, dim_coefficient=4, attention_dropout=0, projection_dropout=0
):
    _, num_patch, channel = x.shape
    assert dim % num_heads == 0
    num_heads = num_heads * dim_coefficient

    x = layers.Dense(dim * dim_coefficient)(x)
    # create tensor [batch_size, num_patches, num_heads, dim*dim_coefficient//num_heads]
    x = tf.reshape(
        x, shape=(-1, num_patch, num_heads, dim * dim_coefficient // num_heads)
    )
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    # a linear layer M_k
    attn = layers.Dense(dim // dim_coefficient)(x)
    # normalize attention map
    attn = layers.Softmax(axis=2)(attn)
    # dobule-normalization
    attn = attn / (1e-9 + tf.reduce_sum(attn, axis=-1, keepdims=True))
    attn = layers.Dropout(attention_dropout)(attn)
    # a linear layer M_v
    x = layers.Dense(dim * dim_coefficient // num_heads)(attn)
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    x = tf.reshape(x, [-1, num_patch, dim * dim_coefficient])
    # a linear layer to project original dim
    x = layers.Dense(dim)(x)
    x = layers.Dropout(projection_dropout)(x)
    return x
# Implement the MLP block
def mlp(x, embedding_dim, mlp_dim, drop_rate=0.2):
    x = layers.Dense(mlp_dim, activation=tf.nn.gelu)(x)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Dense(embedding_dim)(x)
    x = layers.Dropout(drop_rate)(x)
    return x
#Implement the Transformer block
def transformer_encoder(
    x,
    embedding_dim,
    mlp_dim,
    num_heads,
    dim_coefficient,
    attention_dropout,
    projection_dropout,
    attention_type="external_attention",
):
    residual_1 = x
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    if attention_type == "external_attention":
        x = external_attention(
            x,
            embedding_dim,
            num_heads,
            dim_coefficient,
            attention_dropout,
            projection_dropout,
        )
    elif attention_type == "self_attention":
        x = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim, dropout=attention_dropout
        )(x, x)
    x = layers.add([x, residual_1])
    residual_2 = x
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = mlp(x, embedding_dim, mlp_dim)
    x = layers.add([x, residual_2])
    return x

# Implement the EANet model
def get_model(attention_type="external_attention"):
    inputs = layers.Input(shape=input_shape)
    # Image augment
#     x = data_augmentation(inputs)
    # Extract patches.
    x = PatchExtract(patch_size)(inputs)
    # Create patch embedding.
    x = PatchEmbedding(num_patches, embedding_dim)(x)
    # Create Transformer block.
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(
            x,
            embedding_dim,
            mlp_dim,
            num_heads,
            dim_coefficient,
            attention_dropout,
            projection_dropout,
            attention_type,
        )

    x = layers.GlobalAvgPool1D()(x)
    outputs = layers.Dense(32, activation="softmax")(x)
    return outputs


from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense

g = ratings.map(lambda x: x["cut_genre"])
resize = (128, 128)


class MovieModel(tf.keras.Model):

    def __init__(self, use_texts, use_images):
        super().__init__()
        self._use_texts = use_texts
        self._use_images = use_images
        max_tokens = 50_000

        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_place_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_place_ids) + 1, 32)
        ])
        if use_texts:
            self.title_vectorizer = tf.keras.layers.TextVectorization(
                max_tokens=max_tokens)
            self.title_text_embedding = tf.keras.Sequential([
                self.title_vectorizer,
                tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
                tf.keras.layers.GlobalAveragePooling1D(),
            ])
            self.title_vectorizer.adapt(g)
        if use_images:
            self.image_embedding = get_model()

    def call(self, inputs):
        if not self._use_texts:
            if not self._use_images:
                return self.title_embedding(inputs['title'])
            elif self._use_images:
                return tf.concat([
                    self.title_embedding(inputs['title']),
                    self.image_embedding(inputs['image'])], axis=1)
        elif self._use_texts:
            if not self._use_images:
                return tf.concat([
                    self.title_embedding(inputs['title']),
                    self.title_text_embedding(inputs['cut_genre'])], axis=1)
            if self._use_images:
                return tf.concat([
                    self.title_embedding(inputs['title']),
                    self.title_text_embedding(inputs['cut_genre']),
                    self.image_embedding(inputs['image'])], axis=1)

import tensorflow_recommenders as tfrs
class MovielensModel(tfrs.models.Model):

    def __init__(self,use_texts,use_images):
        super().__init__()
        self.query_model = user_model
        self.candidate_model = tf.keras.Sequential([
          MovieModel(use_texts,use_images),
          tf.keras.layers.Dense(32)
        ])
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=places.batch(1000).map(self.candidate_model),
            ),
        )

    def compute_loss(self, features, training=False):

        query_embeddings = self.query_model(features["str_user_id"])
        movie_embeddings = self.candidate_model({'title':features["title"],
                                                 'cut_genre':features["cut_genre"],
                                                  'image':features["image"]})
        return self.task(query_embeddings, movie_embeddings)

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
train = shuffled.take(50525)
test = shuffled.skip(50525).take(12600)
print(len(train))
print(len(test))
cached_train = train.shuffle(100_000).batch(10).cache()
cached_test = test.batch(10).cache()
print(type(cached_train))

# Create a retrieval model.
model = MovielensModel(use_texts=True,use_images=True)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

# Train for 3 epochs.
model.fit(cached_train, epochs=15)
model.evaluate(cached_test, return_dict=True)