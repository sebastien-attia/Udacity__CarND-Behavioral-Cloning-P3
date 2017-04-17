import os
import csv
import cv2
import numpy as np
import sklearn
from scipy.ndimage.interpolation import rotate, shift
from keras.models import load_model

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def build_LeNet_model(drop_out):
    from keras.models import Sequential
    from keras import optimizers
    from keras.layers.core import Dense, Activation, Flatten, Dropout
    from keras.layers import Cropping2D
    from keras.layers.convolutional import Conv2D
    from keras.layers.pooling import MaxPooling2D
    from keras.layers import Lambda

    model = Sequential()
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Conv2D(6, (5, 5), activation="relu"))
    model.add(Dropout(drop_out))
    model.add(MaxPooling2D())
    model.add(Dropout(drop_out))
    model.add(Conv2D(6, (5, 5), activation="relu"))
    model.add(Dropout(drop_out))
    model.add(MaxPooling2D())
    model.add(Dropout(drop_out))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(drop_out))
    model.add(Dense(84))
    model.add(Dropout(drop_out))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def build_nvidia_model(drop_out):
    from keras.models import Sequential
    from keras import optimizers
    from keras.layers.core import Dense, Activation, Flatten, Dropout
    from keras.layers import Cropping2D
    from keras.layers.convolutional import Conv2D
    from keras.layers.pooling import MaxPooling2D
    from keras.layers import Lambda

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((65,20), (0,0))))
    model.add(Conv2D(24, (5,5), strides=(2,2), activation="relu"))
    model.add(Dropout(drop_out))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation="relu"))
    model.add(Dropout(drop_out))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation="relu"))
    model.add(Dropout(drop_out))
    model.add(Conv2D(64, (3,3), activation="relu"))
    model.add(Dropout(drop_out))
    model.add(Conv2D(64, (3,3), activation="relu"))
    model.add(Dropout(drop_out))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(drop_out))
    model.add(Dense(50))
    model.add(Dropout(drop_out))
    model.add(Dense(10))
    model.add(Dropout(drop_out))
    model.add(Dense(1))

    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer='adam')
    return model

def _build_fn(filename, ext="h5"):
    return "%s.%s" % (filename, ext)

class ModelPlayer:
    def __init__(self, correction_steering=0.25, nb_epochs=32, batch_size=128):
        self.correction_steering = correction_steering
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size

    def _build_samples(self, data_set):
        fieldnames = {
            "center": lambda x: x,
            "left": lambda x: x+self.correction_steering,
            "right": lambda x: x-self.correction_steering
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
        np.random.shuffle(samples)
        return samples

    def _generator(self, samples, batch_size=32):
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

    def _plot_history(self, run_name, history_object):
        import matplotlib.pyplot as plt

        ### plot the training and validation loss for each epoch
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        # plt.show()
        plt.savefig( _build_fn(run_name, "png") )
        plt.close()

    def fit(self, run_name, model, data_set, test_size=0.2, initial_epoch=0):
        from sklearn.model_selection import train_test_split
        train_samples, validation_samples = train_test_split(self._build_samples(data_set), test_size=test_size)

        print("Len train samples: ", len(train_samples))
        print("Len validation samples: ", len(validation_samples))

        # compile and train the model using the generator function
        train_generator = self._generator(train_samples, batch_size=self.batch_size)
        validation_generator = self._generator(validation_samples, batch_size=self.batch_size)

        history_object = model.fit_generator(
            train_generator, len(train_samples)/self.batch_size,
            validation_data=validation_generator, validation_steps=len(validation_samples)/self.batch_size,
            epochs=(initial_epoch+self.nb_epochs), verbose=1,
            workers=1, max_q_size=10000,
            initial_epoch=initial_epoch
        )

        model.save( _build_fn(run_name) )

        self._plot_history(run_name, history_object)
        return (initial_epoch + self.nb_epochs)


def _flip_images(img, steering):
    return np.fliplr(img), -steering

def _random_brightness(img, steering):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    rand = np.random.uniform(0.3, 1.0)
    hsv[:,:,2] = rand*hsv[:,:,2]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), steering

def _rotate(img, steering):
    angle = np.random.randint(-15, 15)
    return rotate(img, angle, reshape=False), steering

initial_data0 = ["data", "data_rev",]
# initial_data1 = ["data_std", ]
initial_data1 = ["data_r6",]
# initial_data3 = ["data_r5",]
# initial_data4 = ["data_r4", "data_r3", "data_r2",]
initial_data2 = ["data_play11", "data_play12",]
initial_data3 = ["data_play21",]

data_set_list = (
    (
        (initial_data0, None),
        (initial_data0, _flip_images),
    ),
    (
        (initial_data1, None),
        (initial_data1, _flip_images),
    ),
    (
        (initial_data2, None),
        (initial_data2, _flip_images),
    ),
    #(
    #    (initial_data0, _random_brightness),
    #    (initial_data1, _random_brightness),
    #),
    #(
    #    (initial_data3, None),
    #    (initial_data3, _flip_images),
    #),
    #(
    #    (initial_data, _rotate),
    #)
)

player = ModelPlayer(nb_epochs=50)

is_first = True
last_epoch = 0

for idx, data_set in enumerate(data_set_list):
    if is_first:
        model = build_LeNet_model(0.8)
        print (model.summary())
        is_first = False
    else:
        model = load_model(_build_fn(filename))

    filename = os.path.join("test9", "run_%s" % (idx))
    last_epoch = player.fit(filename, model, data_set, initial_epoch=last_epoch)
