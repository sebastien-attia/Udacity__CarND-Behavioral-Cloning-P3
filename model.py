import os
import csv
import cv2
import numpy as np
import sklearn

nb_epochs=16
correction_steering=1
batch_size=32

def _flip_images(img, steering):
    return np.fliplr(img), -steering

data_set = (
        (["data", "data_r1", "data_r2", "data_r3"], None),
        (["data", "data_r1", "data_r2", "data_r3"], _flip_images),
    )

def _build_samples(data_set, correction_steering=1):
    fieldnames = {
        "center": lambda x: x,
        "left": lambda x: x+correction_steering,
        "right": lambda x: x-correction_steering
    }

    samples = []
    for source_dirs, transformer in data_set:
        for source_dir in source_dirs:
            with open(os.path.join(source_dir,'driving_log.csv')) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    for fieldname in fieldnames:
                        steering = float(row["steering"])
                        img_path = os.path.join(source_dir, 'IMG', os.path.split(row[fieldname])[1])
                        samples.append( (img_path, fieldnames[fieldname](steering), transformer) )
    return samples

def _generator(samples, batch_size=32):
    nb_samples = len(samples)
    while 1:
        np.random.shuffle(samples)
        for offset in range(0, nb_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steerings = []
            for batch_sample in batch_samples:
                image, steering, transformer = cv2.imread(batch_sample[0]), batch_sample[1], batch_sample[2]
                if transformer is not None:
                    image, steering = transformer(image, steering)

                images.append(image)
                steerings.append(steering)

            X_train = np.array(images)
            y_train = np.array(steerings)
            yield sklearn.utils.shuffle(X_train, y_train)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(_build_samples(data_set, correction_steering=correction_steering), test_size=0.2)

# compile and train the model using the generator function
train_generator = _generator(train_samples, batch_size=batch_size)
validation_generator = _generator(validation_samples, batch_size=batch_size)


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Conv2D(24, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(36, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(48, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

print("Len train samples: ", len(train_samples))
print("Len validation samples: ", len(validation_samples))

history_object = model.fit_generator(
    train_generator, len(train_samples)/batch_size,
    validation_data=validation_generator, validation_steps=len(validation_samples)/batch_size,
    epochs=nb_epochs, verbose=1,
    workers=1, max_q_size=10000
)

model.save('model.h5')

import matplotlib.pyplot as plt

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
