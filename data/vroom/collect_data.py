import multiprocessing
import os
import random
import numpy as np
import pickle
import gzip
from tqdm import tqdm
from vroom import Race, RaceCar, AI


def play(steps_between_offset_change, max_offset, batch_id):

    # create objects
    race = Race(fps=10)
    race.generate_track()
    car = RaceCar(speed=20, turn_radius=3)
    race.set_car(car)
    ai = AI(race, car)

    # Create image/action lists
    images = []
    actions = []

    frame = 0
    while True:

        # Step AI (make action)
        action, new_lap = ai.step()

        # Step environment
        image = race.step()

        # Append image and action to their respective lists
        images.append(image[...,::-1])
        actions.append(action + 1)

        # Add frame to the counter
        frame += 1

        # For AI - change driving offset randomly
        if not frame % steps_between_offset_change:
            new_offset = np.random.randint(max_offset * 2 + 1) - max_offset
            ai.apply_offset(new_offset)

        if new_lap:
            break

    # Create the data object
    data = {'observations': np.array(images, dtype=np.uint8), 'actions': np.array(actions, dtype=np.uint8)}

    # And save it gzipped (gzip makes saed file over 320 times smaller)
    with gzip.open(f'data/{batch_id}.pickle.gz', 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':

    # Settings
    SAMPLES = 10000  # How many sequences to collect
    NO_PROCESSES = 10  # How many instances should play at once.

    # Data to pass to the data collecting process
    # (frames between offset change, max offset from teh road center)
    PROFILES = (
        (50, 40),  # 0
        (100, 40),  # 1
        (200, 20),  # 2
    )

    # Will randomly draw from this set for each process
    # The more of one profile type, the more probably it'll be used
    # Whole set means the distribution in the training data
    PROFILE_LIST = (0, 0, 0, 1, 1, 2)

    # Data folder
    os.makedirs('data', exist_ok=True)

    # Create a list of sequential numbers in range of 1 to how many samples we'll save
    # which we'll use as filenames.
    # Shuffle it to shuffle saved samples
    sample_ids = list(range(SAMPLES))
    random.shuffle(sample_ids)
    sample_ids_pointer = 0

    # Loop The number of times which will save desired number of sentences
    for _ in tqdm(range(SAMPLES // NO_PROCESSES), ascii=True):

        # Draw model type and train a new model for each loop
        simulation_data = PROFILES[random.choice(PROFILE_LIST)]

        # And spawn NO_PROCESSES playing processes
        processes = []
        for _ in range(NO_PROCESSES):

            # We'll use multiprocesing to utilize all of the cores
            p = multiprocessing.Process(target=play, kwargs={
                'steps_between_offset_change': simulation_data[0],
                'max_offset': simulation_data[1],
                'batch_id': sample_ids[sample_ids_pointer],
            }, daemon=True)
            p.start()
            processes.append(p)
            sample_ids_pointer += 1

        # Join all of the processes before continuing
        for i in range(NO_PROCESSES):
            processes[i].join()
