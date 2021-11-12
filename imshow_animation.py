import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
Test file. I couldn't figure out how to make FuncAnimation and imshow play nicely with each other. Reference:
https://stackoverflow.com/questions/17212722/matplotlib-imshow-how-to-animate
"""

fps = 30
nSeconds = 5
snapshots = [np.random.rand(5, 5) for _ in range(nSeconds * fps)]

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(8, 8))

a = snapshots[0]
im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)


def animate_func(i):
    if i % fps == 0:
        print('.', end='')

    im.set_array(snapshots[i])
    return [im]


anim = animation.FuncAnimation(
    fig,
    animate_func,
    frames=nSeconds * fps,
    interval=1000 / fps,  # in ms
)

plt.show()

print('Done!')
