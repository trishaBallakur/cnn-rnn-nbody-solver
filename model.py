# from operator import pos
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, ReLU, InputLayer
from tensorflow.keras import Sequential
from image_generator import run_simulation, pos_to_numpy
import matplotlib.pyplot as plt
import os
import argparse



class CNN(tf.keras.Model):
    def __init__(self, args, num_images, input_size=128*128):
        super(CNN, self).__init__()
        self.batch_size = args.batch_size
        self.input_size = input_size
        self.num_images = num_images
        self.loss_list = []
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # self.embedding_matrix = tf.Variable(tf.random.normal([self.num_images, self.embedding_size], stddev=0.1))
        # self.input = InputLayer(input_shape=(None, 526, 526, 100), batch_size=self.batch_size)
        self.conv_layer1 = Conv2D(32, kernel_size=3, strides=(1,1), padding='same', activation='relu', input_shape=(526,526,100), name='input_layer')
        self.conv_layer2 = Conv2D(32, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv_layer3 = Conv2D(32, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv_layer4 = Conv2D(64, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv_layer5 = Conv2D(64, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv_layer6 = Conv2D(8, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv_layer7 = Conv2D(4, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv_layer8 = Conv2D(1, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        
        self.net = Sequential([
            self.conv_layer1,
            self.conv_layer2,
            self.conv_layer3,
            self.conv_layer4,
            self.conv_layer5,
            self.conv_layer6,
            self.conv_layer7,
            self.conv_layer8
        ])

        self.dense_1 = Dense(526*526)
    
    def call(self, inputs):
        # inputs is a [batch_size, image_width, image_height, num_time_steps] tensor

        # conv1_out = self.conv_layer1(inputs)
        # conv2_out = self.conv_layer2(conv1_out)
        # conv3_out = self.conv_layer3(conv2_out)
        # conv4_out = self.conv_layer4(conv3_out)
        # conv5_out = self.conv_layer5(conv4_out)
        out = self.net(inputs)
        
        # f, axs = plt.subplots(2,4)
        # axs = np.reshape(axs, (8,))
        # for i in range(8):
        #     axs[i].imshow(conv4_out[0,:,:,i])
        # #plt.imshow(conv4_out[0,:,:,0])
        # plt.show()
        return out

    def loss(self, predictions, labels):
        diff = predictions - labels  # elementwise for numpy arrays
        m_norm = tf.reduce_sum(abs(diff), axis=None)
        return m_norm / self.batch_size


def generate_data(args, folder="training_data"):
    # remove all data files
    num_simulations = args.generate_data
    for f in os.listdir(folder):
        os.remove(os.path.join(folder, f))

    for sim in range(num_simulations):
        path = "training_data/positions_simulation_" + str(sim) + ".txt"
        # positions is [num_objects, num_dimensions (2), num_time_steps]
        images = run_simulation(args, False, (True, path))
        # images = np.expand_dims(images, axis=0)
        # images = np.transpose(images, (0, 2, 3, 1))
        print("Generated " + path)


def load_data(folder, args):
    fidelity = args.fidelity
    data = []
    num_files = 0
    total_num_files = np.sum([1 for x in os.scandir(folder)])
    for entry in sorted(os.listdir(folder)):
    #for entry in os.scandir(folder):
        num_files += 1
        position_sequence_file = open(folder + "/" + entry, 'r')
        images = np.asarray([pos_to_numpy([[float(value) for value in obj.split()] for obj in line.split(",")], fidelity) for line in position_sequence_file])
        #image_dim = int(fidelity ** 0.5)
        #images = np.reshape(images, (images.shape[0], image_dim, image_dim)).tolist()
        data.append(images)
        print("Loaded data from file: " + str(num_files) + " / " + str(total_num_files))
    
    data = np.array(data)
    data = np.transpose(data, (0, 2, 3, 1))
    print("Data loaded. Data of shape: " + str(data.shape))

    return data

def save_model_weights(model, args):
        """
        Save trained model weights to model_ckpts/

        Inputs:
        - model: Trained VAE model.
        - args: All arguments.
        """
        model_flag = "CNN"
        output_dir = os.path.join("model_ckpts", model_flag)
        output_path = os.path.join(output_dir, model_flag)
        os.makedirs("model_ckpts", exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        model.save_weights(output_path)

def load_weights(model):
    """
    Load the trained model's weights.

    Inputs:
    - model: Your untrained model instance.
    
    Returns:
    - model: Trained model.
    """
    inputs = tf.zeros([1,526,526,100])  # Random data sample
    labels = tf.constant([[1,526,526,1]])
    
    weights_path = os.path.join("model_ckpts", "CNN", "CNN")
    _ = model(inputs)
    model.load_weights(weights_path)
    return model

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

    
    
    for batch_start in range(0, training_data.shape[0], model.batch_size):
        images = training_data[batch_start:batch_start + model.batch_size, :, :, :-1]
        labels = training_data[batch_start:batch_start + model.batch_size, :, :, -1:]
        print("Training batch starting at " + str(batch_start))
        with tf.GradientTape() as tape:
            predictions = model.call(images)
            loss = model.loss(predictions, labels)
            model.loss_list.append(loss.numpy())
            print("Batch loss: " + str(loss.numpy()))

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return model.loss_list

def visualize_intermediate_output(model, x_test, layer_name):
    model = model.net

    layer_name = "conv2d" # feel free to explore other layers
    intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(x_test)

    fig, axes = plt.subplots(8,8)#, figsize=(24,24))
    for i in range(64):
        axes[int(i/8),i%8].imshow(intermediate_output[0,:,:,i])
    plt.show()

def show_animation(data):
    fig = plt.figure(figsize=(4, 5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2, 0])
    for i in range(data.shape[-1]):
        image = data[0,:,:,i]
        plt.sca(ax1)
        plt.cla()
        plt.imshow(image)#, cmap='Greys')
        # plt.scatter(pos[:, 0], pos[:, 1], s=10, color='blue')
        #ax1.set(xlim=(0, len(image)), ylim=(len(image), 0))
        ax1.set_aspect('equal', 'box')
        plt.pause(0.001)


def test(model, args):
    testing_data = load_data("testing_data", args)
    x_test = testing_data[:,:,:,:-1]
    y_test = testing_data[:,:,:,-1:]

    prediction = model.call(x_test)

    visualize_intermediate_output(model, x_test, 'conv2d_7')
    
    # f, (ax1, ax2) = plt.subplots(1,2)
    # ax1.imshow(prediction[0,:,:,0])
    # ax2.imshow(y_test[0,:,:,0])

    # plt.show()
    return prediction, y_test


def parseArguments():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--generate_data", type=int, default=100)
    group.add_argument("--load_data", action="store_true")
    group.add_argument("--load_weights", action="store_true")

    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=10) # implement
    
    
    parser.add_argument("--num_objects", type=int, default=3)
    parser.add_argument("--start_time", type=int, default=0)
    parser.add_argument("--stop_time", type=float, default=1.0)
    parser.add_argument("--time_step_size", type=float, default=0.01)
    parser.add_argument("--fidelity", type=int, default=512)
    
    

    # parser.add_argument("--time_steps", type=int, default=99)
    parser.add_argument("--input_size", type=int, default=28*28)
    
    args = parser.parse_args()
    return args

def main(args):

    if args.load_data:
        training_data = load_data("training_data", args)

        model = CNN(args, 0)
        for epoch in range(5):
            print("Training epoch", epoch)
            
            loss = train(model, training_data)
        
        save_model_weights(model, args)
        print(model.loss_list)
    
    elif args.load_weights:
        # inputs is [batch_size, image_width, image_height, num_time_steps]
        inputs = tf.zeros([1,526,526,100])  # Random data sample
    
        model = CNN(args, 0)
        model = load_weights(model)
        out = test(model, args)
        return out

    elif args.generate_data != None:
        generate_data(args)
        return 0

    else:
        return 0 #show_animation(load_data())



if __name__ == '__main__':
    args = parseArguments()
    main(args)
