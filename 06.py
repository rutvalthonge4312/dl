import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


## Loading images and labels
(train_ds, train_labels), (test_ds, test_labels) = tfds.load("tf_flowers",
 split=["train[:70%]", "train[:30%]"], ## Train test split
 batch_size=-1,
 as_supervised=True, # Include labels
)


train_ds[0].shape


## Resizing images,Image Processing
train_ds = tf.image.resize(train_ds, (150, 150))
test_ds = tf.image.resize(test_ds, (150, 150))

print(train_labels)

train_labels = to_categorical(train_labels, num_classes=5)
test_labels = to_categorical(test_labels, num_classes=5)

print(train_labels[0])


##Load a pre-trained CNN model trained on a large dataset
base_model = VGG16(weights="imagenet", include_top=False, input_shape=train_ds[0].s)
base_model.trainable = False
train_ds = preprocess_input(train_ds)
test_ds = preprocess_input(test_ds)
base_model.summary()




