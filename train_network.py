import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

'''
Class that defines the Tensorflow model specified in Task 1 sub-question C
'''
class MyModel(tf.keras.Sequential):
    def __init__(self):
        super(MyModel, self).__init__()
        self.add(tf.keras.layers.InputLayer(input_shape=(28,28,1)))
        self.add(tf.keras.layers.Conv2D(10, (5, 5)))
        self.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.add(tf.keras.layers.Activation(tf.keras.activations.relu))
        self.add(tf.keras.layers.Conv2D(20, (5, 5)))
        self.add(tf.keras.layers.Dropout(0.5))
        self.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.add(tf.keras.layers.Activation(tf.keras.activations.relu))
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(50))
        self.add(tf.keras.layers.Activation(tf.keras.activations.relu))
        self.add(tf.keras.layers.Dense(10))
        self.add(tf.keras.layers.Activation(tf.keras.activations.softmax))

'''
Arguments - None
Return - Training images, training labels, testing images, testing labels
Description - This function reads the training and testing images and labels for MNSIT data and plots the first 6 examples
'''
def read_and_plot():
    # Read the data
    mnist_dataset = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

    # First 6 examples of training data
    plt.figure(figsize=(1, 1))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap='gray')
        plt.xlabel(train_labels[i])
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=1,hspace=0.1)
    plt.show()
    return train_images,train_labels,test_images,test_labels

'''
Arguments - Training images, training labels, testing images, testing labels
Return - Training images, training labels, testing images, testing labels
Description - This function processes the reads images, processes it and one-hot encodes it for application to model
'''
def preprocess(train_images,train_labels,test_images,test_labels):
    # Make images in range of [0,1]
    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255
    
    # Make images of shape (28,28,1)
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)
    print("train shape", train_images.shape)
    print("test shape", test_images.shape)
    
    # One-hot encoding
    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)
    return train_images,train_labels,test_images,test_labels

'''
Arguments - tensorflow model, training images, training labels, testing images, testing label, #epochs, size of batch
Return - None
Description - This function trains the model and plots the accuracy and loss graphs
'''
def train_and_plot(model,train_images,train_labels,test_images,test_labels,epochs,batch_size):
    # Train the model
    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels), batch_size=batch_size)
    model.summary()

    # Plot training and testing accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy Curve')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.show()
    
    # Plot training and testing loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss Curve')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.show()
    return

'''
Arguments - None
Return - None
Description - This function starts execution and calls function to perform all sub-questions A-E of Task 1
'''
def main():
    tf.random.set_seed(42)
    os.environ['TF_ENABLE_CUDNN'] = '0'
    
    # Read and plot data
    train_images,train_labels,test_images,test_labels = read_and_plot()
    train_images,train_labels,test_images,test_labels = preprocess(train_images,train_labels,test_images,test_labels)
    
    # Create instance of model and compile
    model = MyModel()
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    # Train the model
    epochs = 5
    batch_size = 64
    train_and_plot(model, train_images, train_labels, test_images, test_labels, epochs, batch_size)
    
    # Save weights for future use
    model.save_weights('DNNweights')
    return

if __name__ == "__main__":
   main()
