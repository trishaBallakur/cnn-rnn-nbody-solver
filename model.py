# from operator import pos
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, ReLU
from tensorflow.keras import Sequential
from image_generator import run_simulation, pos_to_numpy
import matplotlib.pyplot as plt
import os

class CNN(tf.keras.Model):
    def __init__(self, num_images, input_size=128*128):
        super(CNN, self).__init__()
        self.batch_size = 10
        self.input_size = input_size
        self.num_images = num_images
        self.loss_list = []

        # self.embedding_matrix = tf.Variable(tf.random.normal([self.num_images, self.embedding_size], stddev=0.1))
        self.conv_layer1 = Conv2D(64, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv_layer2 = Conv2D(32, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv_layer3 = Conv2D(16, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv_layer4 = Conv2D(8, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv_layer5 = Conv2D(1, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        
        self.net = Sequential([
            self.conv_layer1,
            self.conv_layer2,
            self.conv_layer3,
            self.conv_layer4,
            self.conv_layer5
        ])

        self.dense_1 = Dense(526*526)
    
    def call(self, inputs):
        # inputs is a [batch_size, image_width, image_height, num_time_steps] tensor
        # conv1_out = self.conv_layer1(inputs)
        # conv2_out = self.conv_layer2(conv1_out)
        # conv3_out = self.conv_layer3(conv2_out)
        # conv4_out = self.conv_layer4(conv3_out)
        # conv5_out = self.conv_layer5(conv4_out)
        conv5_out = self.net(inputs)
        
        # f, axs = plt.subplots(2,4)
        # axs = np.reshape(axs, (8,))
        # for i in range(8):
        #     axs[i].imshow(conv4_out[0,:,:,i])
        # #plt.imshow(conv4_out[0,:,:,0])
        # plt.show()
        return conv5_out

    def loss(self, predictions, labels):
        diff = predictions - labels  # elementwise for scipy arrays
        m_norm = tf.reduce_sum(abs(diff), axis=None)
        return m_norm / self.batch_size
        

def generate_data(num_simulations, fidelity, folder="training_data"):
    # remove all data files
    for f in os.listdir(folder):
        os.remove(os.path.join(folder, f))

    for sim in range(num_simulations):
        path = "training_data/positions_simulation_" + str(sim) + ".txt"
        # positions is [num_objects, num_dimensions (2), num_time_steps]
        images = run_simulation(False, (True, path), fidelity)
        # images = np.expand_dims(images, axis=0)
        # images = np.transpose(images, (0, 2, 3, 1))
        print("Generated " + path)


def load_data(folder, fidelity):
    data = []
    num_files = 0
    total_num_files = np.sum([1 for x in os.scandir(folder)])
    for entry in os.scandir(folder):
        num_files += 1
        position_sequence_file = open(entry.path, 'r')
        images = np.asarray([pos_to_numpy([[float(value) for value in obj.split()] for obj in line.split(",")], fidelity) for line in position_sequence_file])
        #image_dim = int(fidelity ** 0.5)
        #images = np.reshape(images, (images.shape[0], image_dim, image_dim)).tolist()
        data.append(images)
        print("Loaded data from file: " + str(num_files) + " / " + str(total_num_files))
    
    data = np.array(data)
    data = np.transpose(data, (0, 2, 3, 1))
    print("Data loaded. Data of shape: " + str(data.shape))

    return data

# def save_model_weights(model, args):
#         """
#         Save trained VAE model weights to model_ckpts/

#         Inputs:
#         - model: Trained VAE model.
#         - args: All arguments.
#         """
#         model_flag = "CNN"
#         output_dir = os.path.join("model_ckpts", model_flag)
#         output_path = os.path.join(output_dir, model_flag)
#         os.makedirs("model_ckpts", exist_ok=True)
#         os.makedirs(output_dir, exist_ok=True)
#         model.save_weights(output_path)

# def load_weights(model):
#     """
#     Load the trained model's weights.

#     Inputs:
#     - model: Your untrained model instance.
    
#     Returns:
#     - model: Trained model.
#     """
#     inputs = tf.zeros([1,1,28,28])  # Random data sample
#     labels = tf.constant([[0]])
    
#     weights_path = os.path.join("model_ckpts", "CNN", "CNN")
#     _ = model(inputs)
#     model.load_weights(weights_path)
#     return model

def train(model, training_data):
    # NUM_SIMULATIONS = 100
    # time_sequences_of_images = []
    # for sim in range(NUM_SIMULATIONS):
    #     images = run_simulation(False)
    #     images = np.expand_dims(images, axis=0)
    #     images = np.transpose(images, (0, 2, 3, 1))
    #     time_sequences_of_images.append(images)

    # training data shape is [num_image_sequences, image_width, image_height, num_time_steps]
    
    # want labels to be the image at the final time step for all time sequences

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    for batch_start in range(0, len(training_data), model.batch_size):
        images = training_data[batch_start:batch_start + model.batch_size, :, :, :-1]
        labels = training_data[batch_start:batch_start + model.batch_size, :, :, -1:]
        with tf.GradientTape() as tape:
            predictions = model.call(images)
            loss = model.loss(predictions, labels)
            model.loss_list.append(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return images

def test():
    pass


def main():

    fidelity = 512

    # RUN THIS LINE ONLY FIRST
    training_data = generate_data(100, fidelity)
    # THEN COMMENT OUT LINE YOU JUST RAN AND UNCOMMENT REMAINING LINES OF main()
    # training_data = load_data("training_data", fidelity)

    # model = CNN(0)
    # loss = train(model, training_data)


if __name__ == '__main__':
    main()