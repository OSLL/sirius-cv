import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #to set tf log level
import numpy as np
import tensorflow as tf
from   tensorflow import keras
from   tensorflow.keras import layers
import matplotlib.pyplot as plt
import shutil
from   PIL import Image
from   progressbar import progressbar
import random
from   datetime import datetime #to use time as random seed
from   sklearn.model_selection import train_test_split

#memory limit to fix CUDA Compute bug
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*5)])


def createModel(mean_w, mean_h, num_classes):
    # Model / data parameters
    input_shape = (mean_h, mean_w, 3)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.Dropout(0.5),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.BatchNormalization(),

            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.Dropout(0.5),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.BatchNormalization(),

            layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            layers.Dropout(0.5),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.BatchNormalization(),

            layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
            layers.Dropout(0.5),
            layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.BatchNormalization(),

            layers.Flatten(),
            layers.Dense(4096, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model

def train(model, strategy, batch_size, epochs, doCreateModel=False):
    with strategy:
        x, y, mean_w, mean_h, num_classes = load_dataset("Malaria_cell_dataset")
        std_x = standartize(x, mean_w, mean_h)
        std_x = np.asarray(std_x, dtype=np.float32)
        del x
        std_x /= 255.0
        y = keras.utils.to_categorical(y, num_classes)
        print("Dataset loaded; Image size:", mean_w, "x", mean_h)

        train_x, test_x, train_y, test_y = train_test_split(std_x, y, test_size=0.35)

        if(doCreateModel): model = createModel(mean_w, mean_h, num_classes)

        model.summary()
        try:
            loss = keras.losses.CategoricalCrossentropy()
            opt = keras.optimizers.Adam()
            model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

            os.mkdir('checkpoint.model')
            checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath='checkpoint.model/', save_weights_only=False, save_freq='epoch')
            history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_data=(test_x, test_y), callbacks=[checkpoint_callback])

            score = model.evaluate(test_x, test_y, verbose=0)
            print("Final test accuracy:", score[1])
            print_acc_plot(history)
            name = input("Enter model name: ")
            model.save(name+'.model')
            print("Saved successfully, removing backup...")
            shutil.rmtree('checkpoint.model')
        except KeyboardInterrupt:
            print("Exiting...")
            shutil.rmtree('checkpoint.model')

def load_and_test():
    while(True): #check user's input
        try:
            name = input("Enter model name: ")
            model = keras.models.load_model(name+'.model')
            break
        except OSError:
            print("Model not found, try again!")
    print('\n')
    opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    print("Model loaded")
    do_train = False
    while(True):
        inp = input("Train model? [y/N]")
        if(inp == 'Y' or inp == 'y'):
            do_train = True
            break
        if(inp == 'N' or inp == 'n' or inp == ''): break
    x_train, y_train, x_test, y_test = load_dataset(num_classes)

def print_acc_plot(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def load_dataset(path):
    x = []
    y = []
    #crop params
    sum_w = 0
    sum_h = 0
    img_types = [".jpg", ".png", ".jpeg"]
    label = 0
    classes_count = count_classes(path)
    if(classes_count > 1): print("Loading", classes_count, "classes")
    else: print("Loading", classes_count, "class")
    for directory in os.listdir(path):
        path_to_dir = path + "/" + directory + "/"
        for file in progressbar(os.listdir(path_to_dir)):
            extension = os.path.splitext(file)[1]
            if(extension.lower() not in img_types): continue
            img = Image.open(os.path.join(path_to_dir, file))
            img.load()
            w, h = img.size
            x.append(img)
            y.append(label)
            sum_w = sum_w + w
            sum_h = sum_h + h
        label += 1
    return x, y, int(sum_w/len(x)), int(sum_h/len(x)), classes_count

def count_classes(path):
    counter = 0
    for directory in os.listdir(path):
        counter += 1
    return counter

def standartize(raw_x, mean_w, mean_h):
    x = []
    print("Standartizing...")
    for img in progressbar(raw_x):
        w, h = img.size
        if(mean_h > h):
            new_x = int(w * mean_h/float(h))
            img = img.resize((new_x, mean_h))
        w, h = img.size
        if(mean_w > w):
            new_y = int(h * mean_w/float(w))
            img = img.resize((mean_w, new_y))
        img = randomCrop(img, mean_w, mean_h)
        x.append(np.asarray(img))   
    return x

def randomCrop(img, w_mean, h_mean):
    random.seed(datetime.now())
    w, h = img.size
    w_over = w - w_mean
    h_over = h - h_mean
    w_padding = random.randint(0, w_over)
    h_padding = random.randint(0, h_over)
    img = img.crop((w_padding, h_padding, w_padding+w_mean, h_padding+h_mean))
    return img

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

if(__name__ == "__main__"):
    num_classes = 10
    create = False
    while(True):
        inp = input("Create new model? [y/N]")
        if(inp == 'Y' or inp == 'y'):
            create = True
            break
        if(inp == 'N' or inp == 'n' or inp == ''): break

    if(create):
        train(None, strategy, 32, 15, doCreateModel=True)
    else: load_and_test()