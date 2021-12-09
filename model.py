from numpy.lib.type_check import nan_to_num
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization, Reshape, Conv3D
from tensorflow.keras import Sequential
from image_generator import run_simulation, pos_to_numpy
import matplotlib.pyplot as plt
import os
import argparse
from contextlib import redirect_stdout
from datetime import datetime


model_name = "CNN8"
class CNN8(tf.keras.Model):
    def __init__(self, args):
        super(CNN8, self).__init__()
        self.input_size = args.fidelity * args.fidelity
        self.batch_size = args.batch_size
        self.loss_list = []
        self.num_time_steps = int((args.stop_time - args.start_time) / args.time_step_size)
        
        self.args = args

        self.conv_layer1 = Conv2D(32, kernel_size=3, strides=(1,1), padding='same', activation='relu', input_shape=(args.fidelity,args.fidelity,self.num_time_steps), name='input_layer')
        self.conv_layer2 = Conv2D(32, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv_layer3 = Conv2D(32, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv_layer4 = Conv2D(64, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv_layer5 = Conv2D(64, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv_layer6 = Conv2D(8, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv_layer7 = Conv2D(4, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv_layer8 = Conv2D(1, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.flatten_layer1 = Flatten()      

        self.net = Sequential([
            self.conv_layer1,
            self.conv_layer2,
            self.conv_layer3,
            self.conv_layer4,
            self.conv_layer5,
            self.conv_layer6,
            self.conv_layer7,
            self.conv_layer8,
            self.flatten_layer1
        ])


    
    def call(self, inputs):
        # inputs is a [batch_size, image_width, image_height, num_time_steps] tensor

        out = self.net(inputs)
        
        return tf.cast(out, tf.double)

    def loss(self, predictions, labels):

        cosine_loss = tf.keras.losses.cosine_similarity(labels, predictions, axis=1)
        cosine_loss = tf.reduce_sum(cosine_loss)

        return cosine_loss

def generate_data(args, folder="training_data"):
    # remove all data files
    num_simulations = args.generate_data
    for f in os.listdir(folder):
        os.remove(os.path.join(folder, f))

    for sim in range(num_simulations):
        path = folder + "/positions_simulation_" + str(sim) + ".txt"
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
        - model: Trained CNN model.
        - args: All arguments.
        """
        model_flag = model_name
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
    inputs = tf.zeros([1,model.args.fidelity,model.args.fidelity,model.num_time_steps])  # Random data sample
    
    weights_path = os.path.join("model_ckpts", model_name, model_name)
    _ = model(inputs)
    model.load_weights(weights_path)
    return model

def train(model, training_data):
    # training data shape is [num_image_sequences, image_width, image_height, num_time_steps]
    
    

    # randomly shuffle the inputs
    indices = np.arange(len(training_data))
    np.random.shuffle(indices)
    training_data = tf.gather(training_data, indices)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=model.args.learning_rate)   
    for batch_start in range(0, training_data.shape[0], model.batch_size):

        # want labels to be the image at the final time step for all time sequences
        images = training_data[batch_start:batch_start + model.batch_size, :, :, :-1]
        labels = training_data[batch_start:batch_start + model.batch_size, :, :, -1:]
        
        labels = np.reshape(labels, (labels.shape[0], -1))

        # labels = tf.where(labels == 0, np.ones(labels.shape) * 0.01, labels)

        images = tf.cast(images, tf.double)
        labels = tf.cast(labels, tf.double)

        print("Training batch starting at " + str(batch_start))
        
        with tf.GradientTape() as tape:
            predictions = model.call(images)
            batch_loss = model.loss(predictions, labels)
            model.loss_list.append(batch_loss.numpy())
            print("Batch loss per example: " + str(batch_loss.numpy() / model.batch_size))

        gradients = tape.gradient(batch_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
    save_model_weights(model, model.args)
    
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

def show_animation(data, simulation_num):
    fig = plt.figure(figsize=(4, 5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2, 0])
    for i in range(data.shape[-1]):
        image = data[simulation_num,:,:,i]
        plt.sca(ax1)
        plt.cla()
        plt.imshow(image)#, cmap='Greys')
        # plt.scatter(pos[:, 0], pos[:, 1], s=10, color='blue')
        #ax1.set(xlim=(0, len(image)), ylim=(len(image), 0))
        ax1.set_aspect('equal', 'box')
        plt.pause(0.001)

def similarity(prediction, label):
    prediction = prediction.flatten()
    label = label.flatten()

    similarity = tf.keras.losses.cosine_similarity(label, prediction)

    return similarity

def test(model, args):
    testing_data = load_data("testing_data", args)
    # inputs are images at all timesteps except last
    x_test = testing_data[:,:,:,:-1]
    # labels are image at last timestep
    y_test = testing_data[:,:,:,-1:]

    predictions = None

    """
    Uncomment the following section to see animation
    """
    # x_test = x_test[:10]
    # y_test = y_test[:10]
    # for n in range(50):
    #     predictions = model.call(x_test)

    #     preds = np.reshape(predictions, (predictions.shape[0], args.fidelity, args.fidelity, 1))
        
    #     x_test = tf.concat((x_test[:,:,:,1:], preds), axis=3)

    # # parameters are data, which simulation in the data you want to animate
    # # could also add a breakpoint here and type in the debug console
    # show_animation(x_test, 0)
    
    """
    Uncomment the following section to see outputs of specific layers
    """
    # visualize_intermediate_output(model, x_test, 'conv2d_7')
    
    """
    Uncomment the following section to see predictions vs. labels for all files 
    in ./testing data
    """
    
    filename = "performance_model_" + model_name[-1] + ".txt"

    predictions = model.call(x_test)

    with open(filename, 'w+') as f:
        with redirect_stdout(f):
            model.net.summary()

        f.write("Training examples: 1000" + "\n")
        f.write("Batch size: " + str(args.batch_size) + "\n")
        f.write("Epochs: " + str(args.num_epochs) + "\n")
        f.write("Learning rate: " + str(args.learning_rate) + "\n")     
    
    similarities = np.zeros(len(x_test))
    diffs = np.zeros(len(x_test))
    for i in range(len(x_test)):
        
        prediction = np.reshape(predictions[i], (args.fidelity, args.fidelity))
        label = y_test[i,:,:,0]
        
        if not np.all(prediction == 0):
            prediction = prediction * (1/np.max(prediction))
        if not np.all(label == 0):
            label = label * (1/np.max(label))
        diff = np.abs(prediction - label)
        diffs[i] = np.sum(diff)
        similarities[i] = similarity(prediction, label)
        print("Similarity: " + str(similarities[i]))

        if i < 10:
            # f, (ax1, ax2, ax3) = plt.subplots(1,3)
            f, (ax1, ax2) = plt.subplots(1,2)
            ax1.imshow(prediction)
            ax2.imshow(label)
            # ax3.imshow(diff)
            ax1.set_title("Prediction")
            ax2.set_title("Actual")
            # ax3.set_title("Difference")

            plt.show()

    with open(filename, 'a') as f:
        f.write("Average accuracy over 100 testing sequences: " + str(np.average(similarities)) + "\n")
        f.write("Average difference over 100 testing sequences: " + str(np.average(diffs)))
    

    print("Average accuracy over 100 testing sequences: " + str(np.average(similarities)))
    print("Average difference over 100 testing sequences: " + str(np.average(diffs)))
    return predictions, y_test


def parseArguments():
    parser = argparse.ArgumentParser()

    # want to keep stages separate so we can save progress
    group = parser.add_mutually_exclusive_group()

    # if it should generate data to save in ./training_data folder and how many simulations
    group.add_argument("--generate_data", type=int, default=100)
    
    # if it should train on the data in ./training_data
    group.add_argument("--load_data", action="store_true")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.001) 

    # if it should load the saved weights and test with files in ./testing_data
    group.add_argument("--load_weights", action="store_true")
    
    # other arguments
    parser.add_argument("--num_objects", type=int, default=3)
    parser.add_argument("--start_time", type=int, default=0)
    parser.add_argument("--stop_time", type=float, default=1.0)
    parser.add_argument("--time_step_size", type=float, default=0.01)
    parser.add_argument("--fidelity", type=int, default=128)
    
    args = parser.parse_args()
    return args

def main(args):

    # if program called with --load_data flag
    if args.load_data:
        training_data = load_data("training_data", args)
    
        model = CNN8(args)
        start_time = datetime.now()
        for epoch in range(args.num_epochs):
            print("Training epoch", epoch)
            
            loss = train(model, training_data)
        
        save_model_weights(model, args)
        end_time = datetime.now()
        print(model.loss_list)
        print('Trained in: {}'.format(end_time - start_time))
    
    # if program called with --load_weights flag
    elif args.load_weights:
        # inputs is [batch_size, image_width, image_height, num_time_steps]
        inputs = tf.zeros([1, args.fidelity, args.fidelity, int((args.stop_time - args.start_time) / args.time_step_size)])  # Random data sample
    
        model = CNN8(args)
        model = load_weights(model)
        out = test(model, args)
        return out

    # if program called with --generate_data [num] argument
    elif args.generate_data != None:
        generate_data(args, "training_data")
        return 0

    else:
        return 0 #show_animation(load_data())



if __name__ == '__main__':
    args = parseArguments()
    main(args)
