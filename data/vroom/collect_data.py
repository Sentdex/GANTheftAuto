import multiprocessing
import os
import random
import numpy as np
import pickle
import gzip
import math
from tqdm import tqdm
from vroom import Race, RaceCar, AI


def debug(message, debug_log, console_lock, verbose=False):

    if not debug_log:
        return

    with console_lock, open('./debug_log.txt', 'a') as f:
        if verbose:
            print(message)
        f.write(message + '\n')


def play(steps_between_offset_change, max_offset, c1, c2, c3, c4, reversed, uturns, frame_min, frame_limit, batch_id, data_folder, debug_log, console_lock):

    debug(f' {batch_id:> 8d} - env: starting', debug_log, console_lock)

    data_collected = False
    while not data_collected:

        # create objects
        race = Race(fps=10)
        race.generate_track()
        car = RaceCar(speed=20, turn_radius=3)
        race.set_car(car, reversed=reversed)
        ai = AI(race, car)

        # Create image/action lists
        images = []
        actions = []

        frame = 0
        next_flip = np.random.randint(c3, c4)
        while True:

            # Step AI (make action)
            action, new_lap = ai.step(reversed=reversed)

            # Step environment
            image = race.step()

            # If car is out of track - rstart similation
            if image is None:
                debug(f' {batch_id:> 8d} - env: out of track, restarting', debug_log, console_lock)
                break

            # Append image and action to their respective lists
            images.append(image[...,::-1])
            actions.append(action + 1)

            # Add frame to the counter
            frame += 1

            # For AI - change driving offset randomly
            if not frame % steps_between_offset_change:
                new_offset = np.random.randint(max_offset * 2 + 1) - max_offset
                ai.apply_offset(new_offset)

            # Change driving direction randomly
            if uturns and not frame % next_flip:
                reversed = not reversed
                next_flip = (np.random.randint(c1, c2) if reversed else np.random.randint(c3, c4))

            # If lap finished or number of frames rached - break
            if (not uturns and new_lap) or frame >= frame_limit:

                # Is the number of frames enough?
                if frame < frame_min:
                    debug(f' {batch_id:> 8d} - env: too few frames ({frame}), restarting', debug_log, console_lock)
                    break

                # It is, return
                data_collected = True
                debug(f' {batch_id:> 8d} - env: collected {frame} frames', debug_log, console_lock)
                break

    # Create the data object
    data = {'observations': np.array(images, dtype=np.uint8), 'actions': np.array(actions, dtype=np.uint8)}

    # And save it gzipped (gzip makes saed file over 320 times smaller)
    with gzip.open(f'{data_folder}/{batch_id}.pickle.gz', 'wb') as f:
        pickle.dump(data, f)

    debug(f' {batch_id:> 8d} - env: saved {data_folder}/{batch_id}.pickle.gz', debug_log, console_lock)


if __name__ == '__main__':

    # Settings
    SAMPLES = 60000  # How many sequences to collect
    NO_PROCESSES = 30  # How many instances should play at once.

    # Data to pass to the data collecting process
    # (frames between offset change, max offset from teh road center)
    PROFILES = (
        (50, 40, 15, 50, 50, 100),  # 0
        (100, 40, 50, 100, 150, 200),  # 1
        (200, 20, 100, 150, 250, 300),  # 2
    )

    # Will randomly draw from this set for each process
    # The more of one profile type, the more probably it'll be used
    # Whole set means the distribution in the training data
    PROFILE_LIST = (0, 0, 0, 1, 1, 2)

    # A chance for car to drive the other way
    REVERSE_CHANCE = 0.5

    # A chance of car to make U-turns and circles in this sequence
    UTURN_CHANCE = 0.5

    # A total frame limit (it could drive very long with U-turns enabled)
    FRAME_MIN = 1500
    FRAME_LIMIT = 3000

    # A folder name to save data in:
    DATA_FOLDER = 'data'  # 'data_40k_uturns'

    # Data folder
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # Create a list of sequential numbers in range of 1 to how many samples we'll save
    # which we'll use as filenames.
    # Shuffle it to shuffle saved samples
    sample_ids = list(range(SAMPLES))
    random.shuffle(sample_ids)
    sample_ids_pointer = 0

    # For logging
    console_lock = multiprocessing.Lock()
    debug_log = True
    debug('\n\nStarting data generation', debug_log, console_lock, verbose=True)

    # Validate if created files meet expectations (useful after code update)
    debug('Validating existing files', debug_log, console_lock, verbose=True)
    # Enumerate all of the files
    for file in tqdm(os.listdir(DATA_FOLDER), ascii=True):
        # Path and batch id
        path = f'{DATA_FOLDER}/{file}'
        batch_id = int(file.split('.')[0])
        try:
            # Try to open, ungzip and unpickle the data
            with gzip.open(path, 'rb') as f:
                data = pickle.load(f)
                # Check number of frames in a sequence
                if len(data['observations']) < FRAME_MIN:
                    raise Exception(f'too few frames: {len(data["actions"])}')
                #File is ok, remove batch_id from teh list of ids to prepare
                if batch_id in sample_ids:
                    sample_ids.remove(batch_id)
                debug(f' {batch_id:> 8d} - val: ok', debug_log, console_lock, verbose=False)
        except Exception as e:
            debug(f' {batch_id:> 8d} - val: {e}', debug_log, console_lock)
            # Somethign is wrong - remove the file
            os.unlink(path)

    debug('Generating remaining files', debug_log, console_lock, verbose=True)
    # Loop The number of times which will save desired number of sentences
    for _ in tqdm(range(math.ceil(len(sample_ids) / NO_PROCESSES)), ascii=True):

        # Draw model type and train a new model for each loop
        simulation_data = PROFILES[random.choice(PROFILE_LIST)]

        # And spawn NO_PROCESSES playing processes
        processes = []
        for _ in range(NO_PROCESSES):

            # All done?
            if sample_ids_pointer >= len(sample_ids):
                break

            debug(f' {sample_ids[sample_ids_pointer]:> 8d} - sys: spawning process', debug_log, console_lock)

            # We'll use multiprocesing to utilize all of the cores
            p = multiprocessing.Process(target=play, kwargs={
                'steps_between_offset_change': simulation_data[0],
                'max_offset': simulation_data[1],
                'c1': simulation_data[2],
                'c2': simulation_data[3],
                'c3': simulation_data[4],
                'c4': simulation_data[5],
                'reversed': np.random.random() < REVERSE_CHANCE,
                'uturns': np.random.random() < UTURN_CHANCE,
                'frame_min': FRAME_MIN,
                'frame_limit': FRAME_LIMIT,
                'batch_id': sample_ids[sample_ids_pointer],
                'data_folder': DATA_FOLDER,
                'debug_log': debug_log,
                'console_lock': console_lock
            }, daemon=True)
            p.start()
            processes.append([sample_ids[sample_ids_pointer], p])
            sample_ids_pointer += 1

        # Join all of the processes before continuing
        for i in range(len(processes)):
            debug(f' {processes[i][0]:> 8d} - sys: joining process', debug_log, console_lock)
            processes[i][1].join()
            debug(f' {processes[i][0]:> 8d} - sys: process finished', debug_log, console_lock)
