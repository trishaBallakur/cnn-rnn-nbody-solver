import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.lib.npyio import save
from nbody import getAcc, getEnergy


def dropoff_func(radius, dev):
    """

    :param radius: distance from center
    :param dev: how "spread out" it is
    :return:
    """
    top = - np.power(radius, 4)
    top = np.divide(top, 2 * np.power(dev, 2))
    return np.exp(top)


def position_screen(screen_size):
    if screen_size % 2 == 0:
        print("screen_size should be an odd number, adding 1")
        screen_size += 1
    dev = 0.1
    center = int(screen_size / 2)
    screen = np.zeros((screen_size, screen_size))
    for i in range(screen_size):
        for j in range(screen_size):
            x_2 = np.power((i - center) / screen_size, 2)
            y_2 = np.power((j - center) / screen_size, 2)
            radius = np.sqrt(x_2 + y_2)
            screen[i, j] = dropoff_func(radius, dev)
    return screen



def pos_to_numpy(positions, fidelity, screen=None):
    """

    :param positions: Numpy array of shape (num_particles, 2) that contains particles' x and y positions
    :param fidelity: How detailed the image is (how many pixels along x and y axes)
    :param screen: The representation of the particle in the image.
    Specify to (maybe) speed things up. Otherwise will generate new screen based on value of fidelity
    :return: Image representation of particles' positions
    """
    if screen is None:
        # We want the screen's height (and width) to be around 1.5% of the total image's height
        half_size = int(fidelity * 0.015)
        screen = position_screen(half_size * 2 + 1)
    else:
        half_size = np.size(screen, axis=0)

    image = np.zeros((fidelity + 2 * half_size, fidelity + 2 * half_size))

    for p in range(len(positions)):
        pos = positions[p]
        x_idx = int(pos[0] * fidelity)
        y_idx = int(pos[1] * fidelity)

        # x_idx = 0
        # y_idx = 0

        min_x_idx = x_idx - half_size
        max_x_idx = x_idx + half_size
        min_y_idx = y_idx - half_size
        max_y_idx = y_idx + half_size

        """
        First line doesn't account for padding, second line should
        """
        # image[(x_idx - half_size):(x_idx + half_size + 1), (y_idx - half_size):(y_idx + half_size + 1)] = screen
        
        image_dim = image.shape[0]
        max_image_idx = image_dim - 1
        screen_dim = screen.shape[0]

        # if any part of the object is not in view
        #if min_x_idx < 0 or max_x_idx > max_image_idx or min_y_idx < 0 or max_y_idx > max_image_idx:

            # if some of the object is in view

        # if only right edge is in view
        if max_x_idx >= 0 and min_x_idx < 0:
            lower_x_bound = 0
            upper_x_bound = max_x_idx + 1

        # if only left edge is in view
        elif max_x_idx > max_image_idx and min_x_idx <= max_image_idx:
            lower_x_bound = min_x_idx
            upper_x_bound = max_image_idx + 1
        
        else:
            lower_x_bound = min_x_idx
            upper_x_bound = max_x_idx + 1

        
        # if only bottom edge is in view
        if max_y_idx >= 0 and min_y_idx < 0: 
            lower_y_bound = 0
            upper_y_bound = max_y_idx + 1

        # if only top edge is in view
        elif min_y_idx <= max_image_idx and max_y_idx > max_image_idx:
            lower_y_bound = min_y_idx
            upper_y_bound = max_image_idx + 1

        else:
            lower_y_bound = min_y_idx
            upper_y_bound = max_y_idx + 1

        # still some issues here
        if not (upper_y_bound < 1 or lower_y_bound > max_image_idx or upper_x_bound < 1 or lower_x_bound > max_image_idx): 
            image[lower_x_bound:upper_x_bound, lower_y_bound:upper_y_bound] = screen[screen_dim - (upper_x_bound-lower_x_bound):, screen_dim - (upper_y_bound-lower_y_bound):]

    return image



def write_pos_to_file(pos_save, N, filename):
    
    # erase file
    data_file = open(filename, "w+")    
    data_file.close()

    # open file to write
    data_file = open(filename, "a+")
    
    # for each timestep
    for t in range(pos_save.shape[-1]):
        # for each object
        for n in range(N-1):
            # for each of the 2 dimensions
            for x in range(pos_save.shape[1]):
                # write the coordinate followed by a space
                data_file.write(str(pos_save[n,x,t]) + " ")
            # in between each object put a comma
            data_file.write(", ")
        # write the coords of the last object followed by a new line
        for x in range(pos_save.shape[1]):
            data_file.write(str(pos_save[-1,x,t]) + " ")
        data_file.write('\n')
        # close the file
    data_file.close()

def write_images_to_file(images, filename):
    data_file = open(filename, "w+")    
    data_file.close()
    data_file = open(filename, "a+")

    for i in range(images.shape[0]):
        flattened_image = images[i].flatten()
        # np.reshape(images[i], (images[i][0], -1))
        
        data_file.write(str(flattened_image.tolist()) + "\n")


    data_file.close()





def run_simulation(args, show_plot=False, save_data=(False, "")):
    """ N-body simulation """
    file_path = save_data[1]
    save_data = save_data[0]

    fidelity = args.fidelity

    # Simulation parameters
    N = args.num_objects  # Number of particles
    t = args.start_time  # current time of the simulation
    tEnd = args.stop_time  # time at which simulation ends
    dt = args.time_step_size  # timestep
    softening = 0.1  # softening length
    G = 1.0  # Newton's Gravitational Constant
    plotRealTime = True  # switch on for plotting as the simulation goes along

    # Generate Initial Conditions
    # np.random.seed(17)  # set the random number generator seed

    mass = 3.0 * np.ones((N, 1)) / N  # total mass of particles is 3
    pos = np.random.rand(N, 3)
    pos[:,2] = 0
    # pos = np.random.randn(N, 3)  # randomly selected positions and velocities
    vel = np.random.randn(N, 3)
    vel[:,2] = 0

    # Convert to Center-of-Mass frame
    vel -= np.mean(mass * vel, 0) / np.mean(mass)

    # calculate initial gravitational accelerations
    acc = getAcc(pos, mass, G, softening)

    # calculate initial energy of system
    KE, PE = getEnergy(pos, vel, mass, G)

    # number of timesteps
    Nt = int(np.ceil(tEnd / dt))

    # save energies, particle orbits for plotting trails
    images = []
    pos_save = np.zeros((N, 3, Nt + 1))
    pos_save[:, :, 0] = pos
    KE_save = np.zeros(Nt + 1)
    KE_save[0] = KE
    PE_save = np.zeros(Nt + 1)
    PE_save[0] = PE
    t_all = np.arange(Nt + 1) * dt

    # prep figure
    if show_plot:
        fig = plt.figure(figsize=(4, 5), dpi=80)
        grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
        ax1 = plt.subplot(grid[0:2, 0])
        ax2 = plt.subplot(grid[2, 0])

    
    # # erase data
    # if save_data:
    #     data_file = open(file_path, "w+")    
    #     data_file.close()

    # Simulation Main Loop
    for i in range(Nt):
        
        # if save_data:
            # write_pos_to_file(pos, N, file_path)

        # (1/2) kick
        vel += acc * dt / 2.0

        # drift
        pos += vel * dt

        # update accelerations
        acc = getAcc(pos, mass, G, softening)

        # (1/2) kick
        vel += acc * dt / 2.0

        # update time
        t += dt

        # get energy of system
        KE, PE = getEnergy(pos, vel, mass, G)

        # save energies, positions for plotting trail
        pos_save[:, :, i + 1] = pos
        KE_save[i + 1] = KE
        PE_save[i + 1] = PE

        
        # plot in real time
        if plotRealTime or (i == Nt - 1):
            if show_plot:
                plt.sca(ax1)
                plt.cla()
            xx = pos_save[:, 0, max(i - 50, 0):i + 1]
            yy = pos_save[:, 1, max(i - 50, 0):i + 1]
            # plt.scatter(xx, yy, s=1, color=[.7, .7, 1])

            # discard z position
            image = pos_to_numpy(pos[:,:2], fidelity)
            images.append(image.tolist())
            
            if show_plot:
                plt.imshow(image)#, cmap='Greys')
                # plt.scatter(pos[:, 0], pos[:, 1], s=10, color='blue')
                #ax1.set(xlim=(0, len(image)), ylim=(len(image), 0))
                ax1.set_aspect('equal', 'box')
                ax1.set_xticks(np.arange(0, len(image), (len(image) - len(image)%10)//5))
                ax1.set_yticks(np.arange(0, len(image), (len(image) - len(image)%10)//5))

                plt.sca(ax2)
                plt.cla()
                plt.scatter(t_all, KE_save, color='red', s=1, label='KE' if i == Nt - 1 else "")
                plt.scatter(t_all, PE_save, color='blue', s=1, label='PE' if i == Nt - 1 else "")
                plt.scatter(t_all, KE_save + PE_save, color='black', s=1, label='Etot' if i == Nt - 1 else "")
                ax2.set(xlim=(0, tEnd), ylim=(-300, 300))
                ax2.set_aspect(0.007)

                plt.pause(0.001)

    

    if show_plot:

        # add labels/legend
        plt.sca(ax2)
        plt.xlabel('time')
        plt.ylabel('energy')
        ax2.legend(loc='upper right')

        # Save figure
        plt.savefig('nbody.png', dpi=240)
        plt.show()

    images = np.array(images)

    if save_data:
        write_pos_to_file(pos_save[:,:2,:], N, file_path)
    #     write_images_to_file(images, file_path)

    return images


def main():
    run_simulation(False)
    # run_simulation(True)


if __name__ == "__main__":
    main()
