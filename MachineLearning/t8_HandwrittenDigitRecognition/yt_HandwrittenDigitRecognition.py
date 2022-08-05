import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os.path


def handwritten_digit_recognition_main():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(accuracy)
    print(loss)

    model.save('digits.model')

    HERE = os.path.dirname(os.path.abspath(__file__))

    for x in range(0,10):
        # 'NoneType' object is not subscriptable ???
        # img = cv.imread(f"digits/{x}.png")[:,:,0]
        image_path = os.path.join(HERE, f'digits/{x}.png')
        img = cv.imread(image_path)[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f'the result is probably: {np.argmax(prediction)}')
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()