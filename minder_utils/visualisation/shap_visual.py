from sklearn.preprocessing import normalize
from minder_utils.scripts.weekly_loader import Weekly_dataloader
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input
import os
import numpy as np
import shap
import tensorflow as tf
from minder_utils.configurations import config
import matplotlib.pyplot as plt

tf.compat.v1.disable_v2_behavior()

os.chdir('../')

# Load the data
loader = Weekly_dataloader(num_days_extended=1)
unlabelled = np.load(os.path.join(loader.previous_unlabelled_data, 'activity.npy')).reshape(-1, 24, 14, 1)
X = np.load(os.path.join(loader.previous_labelled_data, 'activity.npy')).reshape(-1, 24, 14, 1)
y = np.load(os.path.join(loader.previous_labelled_data, 'label.npy')).reshape(-1, )
y[y < 0] = 0
y[y > 0] = 1

# Normalise the data
unlabelled = normalize(unlabelled.reshape(unlabelled.shape[0], -1)).reshape(unlabelled.shape[0], 24, 14, 1)
X = normalize(X.reshape(X.shape[0], -1)).reshape(X.shape[0], 24, 14, 1)

# Load the extractor
input_layer = Input(shape=(24, 14, 1))
encoded = Conv2D(4, (3, 3), activation='relu', padding='same')(input_layer)
encoded = Conv2D(3, (3, 3), activation='relu', padding='same')(encoded)
latent = Flatten(name='latent')(encoded)
decoded = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)
encoder = Model(input_layer, latent)
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X, epochs=50, batch_size=512)

# Initialise classifier
clf = Dense(2, activation='sigmoid')(encoder.output)

model = Model(encoder.input, clf)
for idx, layer in enumerate(model.layers):
    if idx < len(model.layers) - 1:
        layer.trainable = False
model.compile('adam', 'sparse_categorical_crossentropy')
model.fit(X, y, epochs=10, batch_size=10)
# Visualise
background = X[np.random.choice(X.shape[0], 100, replace=False)]
e = shap.DeepExplainer(model, background)

def plot(idx):
    plt.clf()
    shap_values = e.shap_values(X[idx:idx+1])
    shap.image_plot(shap_values, -X[idx:idx+1], labels=['No UTI', 'UTI'], hspace='auto', width=100, xticks=config['activity']['sensors'], show=False)


    plt.gcf().subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.3,)
    plt.tight_layout()
    validation = 'no UTI' if int(y[idx]) == 0 else 'UTI'
    plt.savefig('../results/shap/{}.png'.format(validation))

plot(1)
plot(2)