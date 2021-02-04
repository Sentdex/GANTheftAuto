import multiprocessing
import os
import random
import neat
import numpy as np
import cv2
import pickle
import gzip
import gym
from gym.envs.classic_control import rendering
from tqdm import tqdm


runs_per_net = 2


# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):

        env = gym.make("CartPole-v1")

        observation = env.reset()
        fitness = 0.0
        done = False
        while not done:

            action = np.argmax(net.activate(observation))
            observation, reward, done, info = env.step(action)
            fitness += reward

        fitnesses.append(fitness)

    return np.mean(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def train(model_type='full'):

    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # If full model - expect the mean of 200 with the best of 500
    if model_type == 'full' or model_type == 'nudged':
        config.fitness_threshold = 200.
        config.pop_size = 200
        mean = 200
        best = 500
    # For partially trained model. the mean should be at least 80,
    # but the best model should not exceed 150
    elif model_type == 'partial':
        config.fitness_threshold = 80.
        config.pop_size = 3
        mean = 80
        best = 150
    # For a random model, we set model to None
    # and handle this later
    elif model_type == 'random':
        winner = None
        return
    else:
        raise Exception(f'Unknown model type: {model_type}')

    # Almost the tutorial code: https://www.youtube.com/watch?v=ZC0gMhYhwW0
    # We just do not report training to the console or processes will
    # easily make it clogged
    while True:
        try:
            pop = neat.Population(config)
            stats = neat.StatisticsReporter()
            pop.add_reporter(stats)
            #pop.add_reporter(neat.StdOutReporter(True))
            pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
            winner = pop.run(pe.evaluate)
            assert stats.get_fitness_mean()[-1] >= mean
            assert stats.best_genome().fitness <= best
            break
        except Exception as e:
            #print(e)
            pass

    # If we want this model to be nudged in the middle of the simulation,
    # save nudge parameters
    winner.nudge = None
    if model_type == 'nudged':
        winner.nudge = {
            'min_frame': NUDGE_FRAME_MIN,
            'max_frame': NUDGE_FRAME_MAX,
            'min_duration': NUDGE_FRAME_DURATION_MIN,
            'max_duration': NUDGE_FRAME_DURATION_MAX
        }

    # Return winner model
    return winner


def play(winner, resize, batch_id, min_length):

    # This piece of code disables rendering the cartpole simulation
    # The window will still flash, might stay visible sometimes
    # The purpose of this is to make simulation to run significantly faster,
    # do not raise unnecessary CPU usage and do not spawn multiple windows at once
    rendering.Viewer.__orig_init__ = rendering.Viewer.__init__
    def new_init(self, *args, **kwargs):
        self.isopen = False
        self.__orig_init__(*args, **kwargs)
        self.window.set_visible(visible=False)
    rendering.Viewer.__init__ = new_init

    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # If the random model - create a net class with an activate method
    # which is just a random action
    if winner is None:
        class net:
            pass
        net.activate = lambda _: random.randint(0, 1)
    # Otherwise create NEAT network
    else:
        net = neat.nn.FeedForwardNetwork.create(winner, config)

    # If this model should be nudged, get the nudge farme, the nudge duration and the action
    nudge_frame = None
    nudge_duration = None
    nudge_action = None
    if winner is not None and winner.nudge is not None:
        nudge_frame = random.randint(winner.nudge['min_frame'], winner.nudge['max_frame'])
        nudge_duration = random.randint(winner.nudge['min_duration'], winner.nudge['max_duration'])
        nudge_action = random.randint(0, 1)

    # Loop simulation - we need at least minimum number of frames recorded
    while True:

        # Create the gym environment
        env = gym.make("CartPole-v1")
        observation = env.reset()

        # Create "done" flag and image/action lists
        done = False
        images = []
        actions = []

        # Loop until simulation is not done
        step = 0
        while not done:

            # Render the simulation frame, but return it as a NumPy array
            image = env.render(mode="rgb_array")

            # Resize to the desired resolution of the GameGAN model
            image = cv2.resize(image, resize, interpolation=cv2.INTER_LANCZOS4)

            # Uncomment to show
            #cv2.imshow('test', image[...,::-1])
            #cv2.waitKey(1)

            # Get an action from the NEAT model
            action = np.argmax(net.activate(observation))

            # Append image and action to their respective lists
            images.append(image)
            actions.append(action)

            # Nudge the pole? Only at starting step and for goven number of steps
            # We do not log this override (thus we added model's prediction action to teh list)
            if nudge_frame is not None and step >= nudge_frame and step <= nudge_frame + nudge_duration:
                print('nudged', nudge_frame, nudge_duration)
                action = nudge_action

            # Step the environment
            observation, reward, done, info = env.step(action)

            # Increase the step number
            step += 1

        # Close teh environment - prevents some errors
        env.close()

        # If there are not enough frames, try again
        if len(actions) < min_length:
            continue

        # Otherwise create the data object
        data = {'observations': np.array(images, dtype=np.uint8), 'actions': np.array(actions, dtype=np.uint8)}

        # And save it gzipped (gzip makes saed file over 320 times smaller)
        #np.save(f'data/{batch_id}.npy', data)
        with gzip.open(f'data/{batch_id}.pickle.gz', 'wb') as f:
            pickle.dump(data, f)

        # Data saved, break
        break


if __name__ == '__main__':

    # Settings
    SAMPLES = 40000  # How many sequences to collect
    IMG_SIZE = (64, 64)  # GameGAN image size
    MIN_SEQUENCE_LENGTH = 32  # Sequence lenghts that the GameGAn will train on (slice of the whole single sequence)
    NO_LOOPS = 5  # How many times use a single model
    NO_PROCESSES = 20  # And how many instances should play at once. NO_LOOPS * NO_PROCESSES equals number of plays per model

    # Will randomly draw from this set each run
    # The more of one model type, the more probably it'll be used
    # Whole set means the distribution in the training data
    MODEL_LIST = ('full', 'nudged', 'nudged', 'partial', 'partial', 'partial', 'partial')  # 'random' never reached 18 frames in a sequence

    # If nudged model is in use:
    NUDGE_FRAME_MIN = 200  # Minimum step number when we nudge the stick
    NUDGE_FRAME_MAX = 300  # Maximum step number
    NUDGE_FRAME_DURATION_MIN = 3  # And the duration for which we'll nudge the stick - min...
    NUDGE_FRAME_DURATION_MAX = 10  # ... and max

    # Data folder
    os.makedirs('data', exist_ok=True)

    # Create a list of sequential numbers in range of 1 to how many samples we'll save
    # which we'll use as filenames.
    # Shuffle it to shuffle saved samples
    sample_ids = list(range(SAMPLES))
    random.shuffle(sample_ids)
    sample_ids_pointer = 0

    # Loop The number of times which will save desired number of sentences
    for _ in tqdm(range(SAMPLES // (NO_LOOPS * NO_PROCESSES)), ascii=True):

        # Draw model type and train a new model for each loop
        model_type = random.choice(MODEL_LIST)
        print()
        print(f'Training the model: {model_type}')
        winner = train(model_type)

        # Use this model NO_LOOPS times
        print('Collecting the sequences:')
        for i in range(NO_LOOPS):
            print(f'Loop: {i+1}/{NO_LOOPS}')

            # And spawn NO_PROCESSES playying processes
            processes = []
            for _ in range(NO_PROCESSES):

                # We'll use multiprocesing - threading does ot work with gym
                p = multiprocessing.Process(target=play, kwargs={
                    'winner': winner,
                    'resize': IMG_SIZE,
                    'batch_id': sample_ids[sample_ids_pointer],
                    'min_length': MIN_SEQUENCE_LENGTH
                }, daemon=True)
                p.start()
                processes.append(p)
                sample_ids_pointer += 1

            # Join all of the processes before continuing
            for i in range(NO_PROCESSES):
                processes[i].join()
