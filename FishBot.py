import functools
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras.backend import set_session
from tqdm import tqdm

# GPU Setup
assert len(tf.config.list_physical_devices('GPU')) > 0
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
set_session(sess)

# Hyperparameters #
num_epochs = 40
num_filters = 32
batch_size = 16
learning_rate = 5e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

def img_mod(img):
    # Image modification specifications #
    wd = 360
    ht = 360

    layer = Image.new('RGB', (wd, ht), (0, 0, 0))
    img = img.resize((wd, ht))
    layer.paste(img)
    img = np.array(layer)

    return img


def load_data():
    direc = os.getcwd() + '/_LABELED-FISHES-IN-THE-WILD'
    direc_pos = '/fish/'
    direc_neg = '/negative/'

    train_pos = [(direc_pos+x) for x in os.listdir(direc+direc_pos) if x.endswith('.jpg')]
    pos_labels = [[1] for x in train_pos]
    train_neg = [(direc_neg+x) for x in os.listdir(direc+direc_neg) if x.endswith('.jpg')]
    neg_labels = [[0] for x in train_neg]

    imgs = train_pos + train_neg  # [img_mod(Image.open(direc + x)) for x in traimage resize tensorflowin_pos or train_neg]

    imgs = [img_mod(Image.open(direc+x)) for x in imgs]
    labels = pos_labels + neg_labels

    indices = [x for x in range(len(imgs))]
    random.shuffle(indices)

    imgs = [imgs[x] for x in indices]
    labels = [labels[x] for x in indices]

    imgs = np.array(imgs)
    labels = np.array(labels)

    imgs = (imgs).astype(np.float32)
    labels = (labels).astype(np.float32)

    print("\nData Loading Complete")

    return (imgs, labels)


def cnn_model():
    dense = functools.partial(tf.keras.layers.Dense, activation='relu')
    conv2D = tf.keras.layers.Conv2D
    flatten = tf.keras.layers.Flatten
    pool = tf.keras.layers.MaxPool2D

    model = tf.keras.Sequential([
        conv2D(filters=num_filters, kernel_size=(3, 3), strides=2),
        tf.keras.layers.BatchNormalization(),
        pool((2, 2)),
        conv2D(filters=num_filters * 2, kernel_size=(3, 3), strides=2),
        conv2D(filters=num_filters * 4, kernel_size=(3, 3), strides=2),
        flatten(),
        dense(512),
        dense(1, activation=None)
    ])

    return model


def get_loss(imgs, labels, model, optimizer):
    with tf.GradientTape() as tape:
        logits = model(imgs)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train_model(fname, optimizer=tf.keras.optimizers.Adamax(5e-3)):
    print("\nTraining is beginning.\n")
    (train_images, train_labels) = load_data()
    #train_images = (np.expand_dims(train_images, axis=0)/255.).astype(np.float32)

    print("\nData Initialized")

    model = cnn_model()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    for i in range(num_epochs):

        if hasattr(tqdm, '_instances'): tqdm._instances.clear()  # clear if it exists

        print("\nStarting epoch {}/{}".format(i + 1, num_epochs))
        for idx in tqdm(range(0, train_images.shape[0], batch_size)):
            # First grab a batch of training data and convert the input images to tensors
            (images, labels) = (train_images[idx:idx + batch_size], train_labels[idx:idx + batch_size])

            images = tf.convert_to_tensor(images, dtype=tf.float32)

            get_loss(images, labels, model, optimizer)
    print(model.summary())
    model.save(fname)


#train_model('fish_finder', optimizer=optimizer)

model = tf.keras.models.load_model('fish_finder')

test_images = os.getcwd() + '/_LABELED-FISHES-IN-THE-WILD/fish/'
test_images2 = os.getcwd() + '/_LABELED-FISHES-IN-THE-WILD/negative/'
test1 = [(test_images+x) for x in os.listdir(test_images) if x.endswith('.jpg')]
pos_labels = [1 for x in test1]
test2 = [(test_images2+x) for x in os.listdir(test_images2) if x.endswith('.jpg')]
neg_labels = [0 for x in test2]
test = test1 + test2
labels = pos_labels + neg_labels
indices = [x for x in range(len(labels))]
random.shuffle(indices)
ind = random.randint(0, len(indices)-20)
indices = indices[ind:ind+20]
test = [test[x] for x in indices]
test = np.array([img_mod(Image.open(x)) for x in test])
labels = np.array([labels[x] for x in indices])
test = tf.convert_to_tensor((test/255.).astype(np.float32), dtype=tf.float32)
labels = (labels).astype(np.float32)

# Plots the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 4
num_images = num_rows*num_cols
plt.figure(figsize=(4*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plt.imshow(test[i])
    plt.xlabel("{:0.02f} ({})".format(np.squeeze(model(test)[i]), labels[i]), color=('red' if np.squeeze(model(test)[i]) >= 0 else 'blue'))
    plt.grid(False)

plt.show()
