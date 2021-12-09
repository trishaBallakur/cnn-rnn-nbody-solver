Arguments:
    Either:
        "--generate_data", type=int, default=100
        OR
        "--load_data", action="store_true"

    And then:
        "--batch_size", type=int, default=128
        "--num_epochs", type=int, default=10
        "--time_steps", type=int, default=99
        "--fidelity", type=int, default=512



To generate data:
    "python model.py --generate_data [int: number of simulations to run and store data for]"

    This will make a folder in the directory of model.py called "training_data" and fill it with 
    text representations of particle positions during --time_steps number of time_steps. Each line 
    of the data files is a different time step. Commas separarate objects and each object's 
    coordinates are separated by a space.

Once there are data files in "training_data", to load that data and use it to train the model:
    "python model.py --load_data" and then any of the other arguments you want (not all implemented yet)

    Batch loss closer to -1 is better.

    SAVED WEIGHTS ARE OVERRIDDEN AFTER ALL TRAININGS! MAKE A COPY ELSEWHERE IF YOU WANT TO KEEP A TRAINED MODEL!