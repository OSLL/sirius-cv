import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #to set tf log level
import numpy as np
import tensorflow as tf
from   tensorflow import keras
from   tensorflow.keras import layers
import matplotlib.pyplot as plt
import shutil
from   PIL import Image

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])

def createModel():
    # Model / data parameters
    input_shape = (32, 32, 3)

    x_train, y_train, x_test, y_test = load_dataset(num_classes)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.Dropout(0.2),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.BatchNormalization(),

            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.Dropout(0.2),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.BatchNormalization(),

            layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            layers.Dropout(0.2),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.BatchNormalization(),

            layers.Flatten(),
            layers.Dense(1024, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model

def train(model, batch_size, epochs):
    model.summary()
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    try:
        opt = keras.optimizers.Adam()
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        os.mkdir('checkpoint.model')
        checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath='checkpoint.model/', save_weights_only=False, save_freq='epoch')
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[checkpoint_callback])
        score = model.evaluate(x_test, y_test, verbose=0)
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
    for directory in os.listdir(path):
        path_to_dir = path "/" + directory + "/"
        for file in os.listdir(path_to_dir):
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
    return x, y, int(sum_w/len(x)), int(sum_h/len(x))

def standartize(raw_x, mean_W, mean_h):
    x = []
    for img in raw_x:
        w, h = img.size
        if(mean_h > h):
            new_x = int(w * mena_h/float(h))
            img = img.resize((new_x, mean_h))
        w, h = img.size
        if(mean_w > w):
            new_y = int(h * mean_w/float(w))
            img = img.resize((mean_w, new_y))
        img = randomCrop(img, mean_w, mean_h).resize((int(mean_w/3), int(mean_h/3)))
        x.append(np.asarray(img))
    return x

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
        model = createModel()
        train(model, 32, 15)
    else: load_and_test()