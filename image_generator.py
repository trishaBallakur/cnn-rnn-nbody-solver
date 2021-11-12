import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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

    for pos in positions:
        x_idx = int(pos[0] * fidelity)
        y_idx = int(pos[1] * fidelity)

        """
        First line doesn't account for padding, second line should
        """
        # image[(x_idx - half_size):(x_idx + half_size + 1), (y_idx - half_size):(y_idx + half_size + 1)] = screen
        image[(x_idx):(x_idx + 2 * half_size + 1), (y_idx):(y_idx + 2 * half_size + 1)] = screen

    return image


def _update_anim_func(i, im):
    image = pos_to_numpy(np.random.rand(1, 2), 512)
    im.set_array(image)


def _array_anim_func(i, im, positions):
    print(positions[i])
    image = pos_to_numpy(positions[i], 512)
    im.set_array(image)


def _get_unif_movement():
    movement = np.arange(0, 0.75, 0.01)
    array = np.zeros((1000, 3, 2))
    # array[:, ]
    return array


def animate():
    # _get_unif_movement()

    fps = 1
    nSeconds = 5
    fig = plt.figure(figsize=(8, 8))

    positions = np.array([[0.2, 0.7], [0.5, 0.5], [0.9, 0.3]])
    image = pos_to_numpy(positions, 512)

    im = plt.imshow(image)

    # anim = FuncAnimation(fig, update_anim_func, fargs=(im,), frames=nSeconds * fps, interval=1000 / fps)

    positions = np.random.rand(1000, 3, 2)
    anim = FuncAnimation(fig, _array_anim_func, fargs=(im, positions), frames=nSeconds * fps, interval=1000 / fps)
    plt.show()


def main():
    animate()


if __name__ == "__main__":
    main()
