# Car racing environment
# Inspired by: https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py

import numpy as np
import cv2

PI = np.pi

VERSION = 1.07

# Converts radians to progress (0 to 1 float)
def radians_to_progress(radians):
    progress = radians / (2 * PI)
    if progress < 0:
        progress += 1
    return progress


# Converts progress back to radians
def progress_to_radians(progress):
    if progress > 0.5:
        progress -= 1
    return progress * 2 * PI


# 2D point manipulation class
class Point2D:

    # Coordinates
    x = 0
    y = 0

    # Constructor sets the point coordinates
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Depending on a x, y coordinates (which can be positive or negative with (0, 0) as the middle)
    # returns the position depending on an andle around teh center point
    def get_progress(self):
        return radians_to_progress(np.arctan2(self.y, self.x))

    # Takes the progress and distance from the center to create a point
    @staticmethod
    def from_progress(progress, distance=1):
        return Point2D(np.cos(progress_to_radians(progress)), np.sin(progress_to_radians(progress))) * distance

    # Multiplies 2 points or a point and a numerical value
    @staticmethod
    def multiply(left, right):
        if isinstance(right, float) or isinstance(right, int):
            return Point2D(left.x * right, left.y * right)
        elif isinstance(right, Point2D):
            return Point2D(left.x * right.x, left.y * right.y)

    # Multiplies with itself
    def __mul__(self, right):
        return Point2D.multiply(self, right)

    # Adds 2 points or a point and a numerical value 
    @staticmethod
    def add(left, right):
        if isinstance(right, float) or isinstance(right, int):
            return Point2D(left.x + right, left.y + right)
        elif isinstance(right, Point2D) or isinstance(right, Vector2D):
            return Point2D(left.x + right.x, left.y + right.y)

    # Adds to itself
    def __add__(self, right):
        return Point2D.add(self, right)

    # Subtracts 2 points or a point and a numerical value 
    @staticmethod
    def subtract(left, right):
        if isinstance(right, float) or isinstance(right, int):
            return Point2D(left.x - right, left.y - right)
        elif isinstance(right, Point2D) or isinstance(right, Vector2D):
            return Point2D(left.x - right.x, left.y - right.y)

    # Subtracts from itself
    def __sub__(self, right):
        return Point2D.subtract(self, right)

    # Text representation while printing a point
    def __str__(self):
        return f'Point2D {{{self.x}, {self.y}}}'

    # Text representation while printing a point
    def __repr__(self):
        return self.__str__()

    # Placeholder point on a list m,eaning that there's no other points
    def is_last(self):
        return self.x == -1

    # Returns integer coordinates from float point values
    def as_integer_list(self, opencv=False):
        point = tuple([np.round(self.x).astype(np.int32), np.round(self.y).astype(np.int32)])
        if opencv:
            point = point[::-1]
        return point

    # Moves point along a vector
    def move_on_line(self, vector):
        return self + vector

    # Moves point along an arc
    def move_on_arc(self, current_progress, delta_progress, radius, forward_vector):
        # Get current progress (direction) vector from progress
        direction_vector = Vector2D.from_progress(current_progress)
        # Rotate it by a given progress
        movement_vector = direction_vector.rotate(delta_progress)
        # Add a forward vector and side vector (multiplied by radius) to self
        # side vector (multiplied by radius) - a point after moving on arc (like when a car is turning)
        return self + forward_vector + (direction_vector - movement_vector) * radius

    # Calculate a distance between points
    def distance(self, point):
        return np.sqrt((self.x - point.x)**2 + (self.y - point.y)**2)


# 2D vector manipulation class
class Vector2D:

    # Coordinates, progress (dorection) and length
    x = 0
    y = 0
    progress = np.nan
    length = 0

    # Constructor sets the vector size, progress and length
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.set_progress_and_length()

    # Calculate length and progress from vector data
    def set_progress_and_length(self):
        self.length = np.sqrt(self.x**2 + self.y**2)
        self.progress = radians_to_progress(np.arctan2(self.y, self.x))

    # Multiplies 2 vectors or a vector and a numerical value
    @staticmethod
    def multiply(left, right):
        if isinstance(right, float) or isinstance(right, int):
            return Vector2D(left.x * right, left.y * right)
        elif isinstance(right, Vector2D):
            return Vector2D(left.x * right.x, left.y * right.y)

    # Multiplies with itself
    def __mul__(self, right):
        return Vector2D.multiply(self, right)

    # Adds 2 vectors or a vector and a numerical value
    @staticmethod
    def add(left, right):
        if isinstance(right, float) or isinstance(right, int):
            return Vector2D(left.x + right, left.y + right)
        elif isinstance(right, Vector2D) or isinstance(right, Point2D):
            return Vector2D(left.x + right.x, left.y + right.y)

    # Adds with itself
    def __add__(self, right):
        return Vector2D.add(self, right)

    # Subtracts 2 vectors or a vector and a numerical value
    @staticmethod
    def subtract(left, right):
        if isinstance(right, float) or isinstance(right, int):
            return Vector2D(left.x - right, left.y - right)
        elif isinstance(right, Vector2D) or isinstance(right, Point2D):
            return Vector2D(left.x - right.x, left.y - right.y)

    # Subtracts with itself
    def __sub__(self, right):
        return Vector2D.subtract(self, right)

    # Text representation while printing a vector
    def __str__(self):
        return f'Vector2D {{{self.x}, {self.y}}}'

    # Text representation while printing a vector
    def __repr__(self):
        return self.__str__()

    # Takes a progress and a length to create a vector
    @staticmethod
    def from_progress(progress, length=1):
        return Vector2D(np.cos(progress_to_radians(progress)), np.sin(progress_to_radians(progress))) * length

    # Rotates a vector 90 degrees left
    def rotate_left(self):
        return Vector2D(-self.y, self.x)

    # Rotates a vector 90 degrees right
    def rotate_right(self):
        return Vector2D(self.y, -self.x)

    # Rotates a vector by a given progress
    def rotate(self, progress):
        return self.from_progress(self.progress + progress, self.length)

    # Creates a vector frm 2 points
    @staticmethod
    def from_points(point1, point2):
        return Vector2D(point1.x - point2.x, point1.y - point2.y)

    # Calculates a vector projections
    def get_projection_of(self, vector):
        return self.x * vector.x + self.y * vector.y


# Main race class
class Race:

    # Track color variables
    road_color = np.array([131, 141, 151])
    road_shades = 3
    road_shades_visibility_factor = 15
    road_shades_interval = 3
    grass_color = np.array([50, 200, 50])
    grass_tile_size = 20
    grass_tile_shades = 2
    grass_tile_shades_visibility_factor = 10

    # Inits teh class with a desired FPS
    def __init__(self, fps, debug=False):
        self.fps = fps
        self.debug = debug

    # Generates a new track
    def generate_track(self, key_points=15, track_radius=500, repeats=5, turn_rate=0.015, point_step=3, first_point_shift=20, road_width=30):

        # Key points are main points in a track that the road will follow
        self.key_points = []

        # For each of the point, randomize it's progress around and a distance from the center point,
        # makes point that create a shape looking like an entangement of a star and a prism
        for point in range(key_points):
            key_point_progress = ((point + 0.5 * np.random.uniform()) / key_points) if point > 0 else 0
            key_point_distance = track_radius * (1 if point == 0 or point == key_points - 1 else np.random.uniform(0.25, 1))

            self.key_points.append((key_point_progress, Point2D.from_progress(key_point_progress, key_point_distance)))

        # Count number of points
        key_point_number = len(self.key_points)

        # Adds the first point once again for continuity (only for a direction)
        # and a placeholder "finished" point
        self.key_points.append([1 + self.key_points[0][0], self.key_points[0][1]])
        self.key_points.append([1 + self.key_points[1][0], Point2D(-1, -1)])

        # Starting position and direction - starts ata  start point at a track radius
        current_position, current_direction = Point2D(track_radius, 0), 0

        # Track points are individual track points, they are atomic points making a track
        # The closer they are, the bigger track's resolution
        self.track_points = []

        # We repeat a track creation so the start point and end point match their directions
        # well enough (almost perfectly)
        for repeatition in range(repeats):

            # Destination variables keep an index, progress and point of a next key point
            # so a point that we are currently creating a track to
            destination_index = 0
            destination_progress = None
            destination_point = None

            # Loop over each track point generation - repeats until last key point is reached
            while True:
                current_progress = current_position.get_progress()

                # If we are going towards repeated starting point, progress is going to switch
                # in the middle from 1.0 to 0.0 - keep it above one in this case (needed by some math below)
                if destination_index >= key_point_number and current_progress < 0.2:
                    current_progress += 1

                # If this is the first track point or we just reached some key point
                # increment key point index and set a new destination
                # (follow witha  track creation towards a ew point)
                if destination_progress is None or current_progress > destination_progress or current_position.distance(destination_point) < point_step:
                    destination_index += 1
                    destination_progress, destination_point = self.key_points[destination_index]

                # If we just created the whole track and the next point is a placeholder "end" - break the loop
                if destination_point.is_last():
                    break

                # A vector pointing from the center to "outside" with a current progress
                progress_vector = Vector2D.from_progress(current_direction)
                # A vector perpendicular to the progress vector - shows the general track direction
                direction_vector = progress_vector.rotate_left()
                # A vector pointing from the current position of a track creation towards the next key point
                destination_vector = Vector2D.from_points(destination_point, current_position)
                # Remember the direction from the previous loop
                previous_direction = current_direction

                # Get a destination vector projection on a progress vector - if the difference is above small margin
                # (which helps to avoid fluctuations), turn the track vector towards destination by a turn rate
                difference = progress_vector.get_projection_of(destination_vector)
                absolute_difference = abs(difference)
                if absolute_difference > 0.05:
                    current_direction -= min(turn_rate, 0.001 * absolute_difference) * np.sign(difference)

                # Add a vector made of current track direction vector and a step size to current position
                # to calculate the next track's point
                current_position += direction_vector * point_step

                # If this is the last iteration over whole track creation - save track points to a list
                if repeatition == repeats - 1:
                    # We want to save a track direction vector, not a progress direction vector - rotate it left
                    # track direction is a float containing number of repetitions - discard the whole part,
                    # the decimal part is an actual progress
                    track_direction = (np.mean([previous_direction, current_direction]) + 0.25) % 1
                    # Add a track point to the list
                    self.track_points.append((current_progress, track_direction, current_position))

        # Lets shift the start point a given number of point by moving some initial point to the end of the list
        self.track_points = self.track_points[first_point_shift:] + self.track_points[:first_point_shift]

        # Track image - slightly bigger than the radius (diameter is twice teh radius + some buffer - 3.0)
        self.track_image = np.zeros((int(track_radius * 3.0), int(track_radius * 3.0), 3)).astype(np.uint8)

        # Create the grass tiles of 2 shades
        # Iterate over x and y by self.grass_tile_size steps (tile size)
        for x in range(0, self.track_image.shape[1] - 1, self.grass_tile_size):
            for y in range(0, self.track_image.shape[0] - 1, self.grass_tile_size):
                # Tile
                contour = np.array([[
                    [y, x],
                    [y, x + self.grass_tile_size],
                    [y + self.grass_tile_size, x + self.grass_tile_size],
                    [y + self.grass_tile_size, x]
                ]], dtype=np.int32)

                # Draw a countour, size of -1 means fill it in
                cv2.drawContours(
                    self.track_image,  # image
                    [contour],  # contour made of given number of points
                    -1,  # contour width, -1 means to fill it in
                    self.grass_color * (  # dimm a bit every nth tile
                        1 - ((x / self.grass_tile_size + y / self.grass_tile_size) % self.grass_tile_shades) / self.grass_tile_shades_visibility_factor
                    ),
                    -1
                )

        # Create road polygons
        for step in range(0, len(self.track_points)):

            # Get a current and a previous road points
            current_point = self.track_points[step]
            previous_point = self.track_points[step - 1]
            # Create road direction vectors from progress values
            current_direction_vector = Vector2D.from_progress(current_point[1], road_width / 2)
            previous_direction_vector = Vector2D.from_progress(previous_point[1], road_width / 2)

            # Create a track's polygon contour by calculating points on a left and a right of both road points
            # This takes road direction into account to calculate right point position
            contour = np.array([[
                (current_point[2] + current_direction_vector.rotate_left() + self.track_image.shape[0] // 2).as_integer_list(opencv=True),
                (current_point[2] + current_direction_vector.rotate_right() + self.track_image.shape[0] // 2).as_integer_list(opencv=True),
                (previous_point[2] + previous_direction_vector.rotate_right() + self.track_image.shape[0] // 2).as_integer_list(opencv=True),
                (previous_point[2] + previous_direction_vector.rotate_left() + self.track_image.shape[0] // 2).as_integer_list(opencv=True)
            ]], dtype=np.int32)

            cv2.drawContours(
                self.track_image,  # image
                [contour],  # contour made of given number of points
                -1,  # contour width, -1 means to fill it in
                self.road_color * (  # dimm a bit every nth polygon
                    1 - ((step // self.road_shades_interval) % self.road_shades) / self.road_shades_visibility_factor),
                -1
            )

        # If debug is enabled, draw road points and road key points
        if self.debug:
            for point in self.key_points:
                point = (point[1] + Point2D(self.track_image.shape[0] // 2, self.track_image.shape[1] // 2)).as_integer_list()
                self.track_image[point[0] - 1:point[0] + 1, point[1] - 1:point[1] + 1] = [0, 255, 0]
            for point in self.track_points:
                point = (point[2] + Point2D(self.track_image.shape[0] // 2, self.track_image.shape[1] // 2)).as_integer_list()
                self.track_image[point] = [0, 0, 255]
            cv2.imshow('Whole track rotated', self.track_image)
            cv2.waitKey(0)

    # Rotate a track to match teh race car along a point in a center between back wheels
    # Slice a given size witha  car set at the bottom center part
    def slice(self, position, direction):

        # A point is always 0-centered, so it can have positive or negative coordinates
        # adding half of the image size translates it to the image coordinates (opencv has coordinates swapped)
        point_opencv = (position + Point2D(self.track_image.shape[0] // 2, self.track_image.shape[1] // 2)).as_integer_list(opencv=True)
        point = (position + Point2D(self.track_image.shape[0] // 2, self.track_image.shape[1] // 2)).as_integer_list()

        # If debug is enabled, draw a current car's position on a track
        if self.debug:
            self.track_image[point[0] - 1:point[0] + 1, point[1] - 1:point[1] + 1] = [255, 0, 0]

        # Pre-slice the image - we do not need to rotate the whole image
        track_pre_slice = self.track_image[point[0] - 85:point[0] + 85, point[1] - 85:point[1] + 85]

        # Make sure that we are not out out the track
        if track_pre_slice.shape[0] < 170 or track_pre_slice.shape[1] < 170:
            return None

        # Rotation matrix for the track image, centered on a car's point between back wheels
        # Progress needs to be converted to degrees for this
        rotation_matrix = cv2.getRotationMatrix2D((85, 85), 180 - direction * 360, 1.0)

        # Make actual image rotation
        result = cv2.warpAffine(track_pre_slice, rotation_matrix, (170, 170), flags=cv2.INTER_LINEAR)

        # If debug is enabled, show a track
        if self.debug:
            # Rotation matrix for the track image, centered on a car's point between back wheels
            # Progress needs to be converted to degrees for this
            rotation_matrix = cv2.getRotationMatrix2D(point_opencv, 180 - direction * 360, 1.0)

            # Make actual image rotation
            debug_result = cv2.warpAffine(self.track_image, rotation_matrix, self.track_image.shape[:2], flags=cv2.INTER_LINEAR)

            cv2.imshow('Whole track rotated', debug_result)

        # Get a car mask and image - mask will be used to zero-out the car area (to black)
        # and then whole car is added to teh image (makes black color transparent)
        car, car_mask = self.car.get_car()

        # Slice a 64x64 image with a point, which is a point between car's back wheels
        # placed at the bottom center
        track_slice = result[85 - 56:85 + 8, 85 - 32:85 + 32]
        # Apply car mask to make a car areas black
        track_slice[48:59, 29:35] *= car_mask
        # Add car image
        track_slice[48:59, 29:35] += car

        # If debug is enabled, show the sliced image (enlargened by a factor of 4)
        if self.debug:
            cv2.imshow('Sliced and enlargened image', cv2.resize(track_slice, (256, 256), interpolation=cv2.INTER_NEAREST))
            cv2.waitKey(1)

        # Return a track slice
        return track_slice

    # Add a car to the track
    def set_car(self, car, reversed=False):
        self.car = car
        self.car.set_position_and_rotation(self.track_points[0 if not reversed else -1][2], (self.track_points[0][1] - (0.5 if reversed else 0)) % 1, 0 if not reversed else len(self.track_points) - 1)

    # Apply steering to a car (relays a control to the car's object)
    def apply_steering(self, throttle=0, steering=0):
        self.car.apply_steering(throttle, steering)

    # Step teh simulation
    def step(self, delta_time=None):

        # Time difference between steps (frames)
        if delta_time is None:
            delta_time = 1 / self.fps

        # Get a new car position and rotation after given amount of time
        # taking given controls into consideration
        position, direction = self.car.get_new_position_and_direction(delta_time)

        # With this new position and rotation, make a new track slice and return it
        return self.slice(position, direction)


# Race car class
class RaceCar:

    # Car pixel values
    car = np.array([
        [[  0,   0, 255], [  0,   0, 255], [  0,   0, 255], [  0,   0, 255], [  0,   0, 255], [  0,   0, 255]],
        [[  0,   0,   0], [255, 255, 255], [  0,   0, 255], [  0,   0, 255], [255, 255, 255], [  0,   0,   0]],
        [[  0,   0,   0], [  0,   0,   0], [  0,   0, 255], [  0,   0, 255], [  0,   0,   0], [  0,   0,   0]],
        [[  0,   0,   0], [255, 255, 255], [  0,   0, 255], [  0,   0, 255], [255, 255, 255], [  0,   0,   0]],
        [[255, 255, 255], [255, 255, 255], [  0,   0, 255], [  0,   0, 255], [255, 255, 255], [255, 255, 255]],
        [[255, 255, 255], [  0,   0, 255], [  0,   0, 255], [  0,   0, 255], [  0,   0, 255], [255, 255, 255]],
        [[  0,   0, 255], [  0,   0, 255], [  0,   0, 255], [  0,   0, 255], [  0,   0, 255], [  0,   0, 255]],
        [[  0,   0,   0], [255, 255, 255], [  0,   0, 255], [  0,   0, 255], [255, 255, 255], [  0,   0,   0]],
        [[  0,   0,   0], [  0,   0,   0], [  0,   0, 255], [  0,   0, 255], [  0,   0,   0], [  0,   0,   0]],
        [[  0,   0,   0], [255, 255, 255], [  0,   0, 255], [  0,   0, 255], [255, 255, 255], [  0,   0,   0]],
        [[  0,   0, 255], [  0,   0, 255], [  0,   0, 255], [  0,   0, 255], [  0,   0, 255], [  0,   0, 255]],
    ], dtype=np.uint8)

    # Set car's speed and turn radius
    def __init__(self, speed, turn_radius):
        self.speed = speed
        self.turn_radius = turn_radius
        self.car_animation_step = 0

    # Set a position and rotation (usually will be used to set it to match the first tracks'
    # point when car is placed on a track)
    def set_position_and_rotation(self, position, rotation, starting_point_index):
        self.position = position
        self.rotation = rotation
        self.starting_point_index = starting_point_index

    # Set current steering
    def apply_steering(self, throttle=0, steering=0):
        self.throttle = throttle
        self.steering = steering

    # Returns current car's image
    def get_car(self):

        # Make a copy of a car
        car = self.car.copy()

        # Add a shade on tires to imitate wheels rolling
        if self.car_animation_step % 4:
            car[int(4 - self.car_animation_step % 4), 0] = [80, 80, 80]
            car[int(4 - self.car_animation_step % 4), 5] = [80, 80, 80]
            car[int(10 - self.car_animation_step % 4), 0] = [80, 80, 80]
            car[int(10 - self.car_animation_step % 4), 5] = [80, 80, 80]

        # If steering is applied, update front wheels to reflect this
        if self.steering == 1:
            car[1, 0], car[1, 1] = car[1, 1].copy(), car[1, 0].copy()
            car[3, 4], car[3, 5] = car[3, 5].copy(), car[3, 4].copy()
        elif self.steering == -1:
            car[1, 4], car[1, 5] = car[1, 5].copy(), car[1, 4].copy()
            car[3, 0], car[3, 1] = car[3, 1].copy(), car[3, 0].copy()

        # Calculate the car's mask (while color means trasparent)
        car_mask = (1 - np.expand_dims(np.any(car != [255, 255, 255], axis=2), axis=-1) * [1, 1, 1]).astype(np.uint8)

        # Set while color to black - it's in a mask already and later we'll add this image to teh tracks image - 
        # mask zeroes car pixels, while area should not change them when added
        car *= np.expand_dims(np.any(car != [255, 255, 255], axis=2), axis=-1)

        # Update wheela nimation step counter
        self.car_animation_step += self.throttle

        # Return car and car mask
        return car, car_mask

    # Calculate new car's position and direction given delta time
    def get_new_position_and_direction(self, delta_time):

        # How much car moved this period - speed multiplied by a delta of time and then by a direction (1 or -1)
        movement = self.speed * delta_time * self.throttle

        # Car's forward vector from rotation and distance moved this period
        forward_vector = Vector2D.from_progress(self.rotation, movement)

        # If steering is applied
        if self.steering != 0:

            # Calculate progress on an arc during turning
            delta_arc_progress = radians_to_progress(self.steering * movement / (2 * PI * -self.turn_radius))

            # Calculate new position while moving on an arc (car is turning)
            self.position = self.position.move_on_arc(self.rotation, delta_arc_progress, -self.turn_radius, forward_vector)

            # Update current rotation to reflect rotation added during this period
            # and keep rotation (progress) within 0 to 1 range
            self.rotation += delta_arc_progress
            self.rotation %= 1

        # If steering is not applied
        else:

            # Move car forward on a line following a forward vector
            self.position = self.position.move_on_line(forward_vector)

        # Return new position and rotation
        return self.position, self.rotation


# AI driving class
class AI:

    # Sets a race track and a car objects
    def __init__(self, race, car, debug=False):
        self.race = race
        self.car = car
        self.apply_offset()
        self.debug = debug
        self.current_point_index = car.starting_point_index

        self.original_road_points_array = self.make_road_points_array(self.race.track_points)

    # Steps AI re-calculate current actions
    def step(self, reversed=False, further_point=20, distance_multiplier=0.1, action_trigger_difference=0.05):

        # Find closes road node to teh car
        closest_point_index, closest_point, _ = self.find_closest_road_node_to_car()
        # Calculate which forward nodes to use (depending on a distance and direction)
        road_node_index = int(closest_point_index + np.round(further_point*distance_multiplier) * (-1 if reversed else 1)) % len(self.track_points)
        # Calculate car to road node vector
        car_node_vector = Vector2D.from_points(self.track_points[road_node_index][2], self.car.position)

        # Default action - no action
        action = 0

        # Calculate a progress difference between car's direction and road node's direction
        progress_difference = self.car.rotation - car_node_vector.progress
        # Keep it in -0.5 to 0.5 range
        if progress_difference > 0.5:
            progress_difference -= 1
        if progress_difference < -0.5:
            progress_difference += 1
        # If it's bigger than a small margin - perform an actionto to change car's direction
        if abs(progress_difference) > action_trigger_difference:
            action = np.sign(progress_difference)

        # Apply action as a car steering action
        self.car.apply_steering(throttle=1, steering=action)

        # Lap finished flag
        new_lap = False
        if not reversed and closest_point_index < self.current_point_index and closest_point_index < 20 or \
               reversed and closest_point_index > self.current_point_index and closest_point_index > len(self.track_points) - 20:
            new_lap = True
        self.current_point_index = closest_point_index

        return action, new_lap

    # Finds closest road node to teh car
    def find_closest_road_node_to_car(self):
        return self.find_closest_road_node_to_point(self.car.position, self.track_points, self.track_points_array)

    # Finds closest road node to the point
    def find_closest_road_node_to_point(self, other_point, node_list, node_list_array):

        # Optimized version using NumPy array calculation

        # Create a (2, 1) array with a point in question coordinates
        other_point = np.array([[other_point.x], [other_point.y]])

        # Subtract point in question  from all of the points, square points and sum tehm along 0th axis
        # To get a list of distances from this point
        distances = np.sqrt(np.square(node_list_array - other_point).sum(axis=0))

        # Get an index of a closes point, this point and the distance from point in question
        closest_point_index = np.argmin(distances)
        closest_point = node_list[closest_point_index]
        closest_distance = distances[closest_point_index]

        # Old method
        '''
        # Closest point, it's index and a smallest distance variables
        closest_point = None
        closest_point_index = None
        closest_distance = 99999999

        # For each point in a road point list
        for index, (_, _, point) in enumerate(node_list):

            # Calculate a distance from a point in question
            tested_distance = point.distance(other_point)

            # If teh distance is smaller than the smallest one, set this point as the closest
            if tested_distance < closest_distance:
                closest_distance = tested_distance
                closest_point = point
                closest_point_index = index
        '''

        # Return point data
        return closest_point_index, closest_point, closest_distance

    def make_road_points_array(self, road_points):

        # Create an array of shape (2, len), 2 because of x and y, and len is a length of a road point list
        points_array = np.zeros((2, len(road_points)))

        # Set x and y coordinates from road point list
        for index, (_, _, point) in enumerate(road_points):
            points_array[0][index] = point.x
            points_array[1][index] = point.y

        return points_array

    # Set's an offset form the road's center to drive on
    def apply_offset(self, offset=0):

        # If an offset is 0 - no maths is necessary, just copy points fro the track object
        if offset == 0:
            self.track_points = self.race.track_points.copy()
            self.track_points_array = self.make_road_points_array(self.track_points)
            return

        # List of new points
        self.track_points = []

        # Enumerate all of teh points
        for index, point in enumerate(self.race.track_points):

            # Given point coordonates and progress, calculate an offset coordinates
            new_point = point[2] + Vector2D.from_progress(point[1]).rotate_right() * offset

            # Check if a new point is closer to the road than teh offset - if it is,
            # new point is too close to some other part of teh tyrack - don;t add it
            if self.find_closest_road_node_to_point(new_point, self.race.track_points, self.original_road_points_array)[2] < abs(offset) - .1:
                continue

            # Create a new point using previous data and new coordinates
            self.track_points.append((
                point[0],
                point[1],
                new_point
            ))

        # If there are sharp turns, V-shaped, remove some additional points to allow
        # for smoother turning
        # Iterate al of the new points
        for index, point in enumerate(self.track_points):

            # If point is already removed, continue
            if point is None or self.track_points[index-1] is None:
                continue

            # Calculate the progression difference between current and previous point
            # and keep it in nage -0.5 to 0.5
            progress_difference = point[1] - self.track_points[index-1][1]
            if progress_difference > 0.5:
                progress_difference -= 1
            if progress_difference < -0.5:
                progress_difference += 1

            # If it's bigger than 1/8th
            if abs(progress_difference) > 0.125:

                # Iterate over calculated number of points and remove them
                for i in range(int(self.car.turn_radius * 10 * abs(progress_difference))):
                    self.track_points[(index+i)%len(self.track_points)] = None
                    self.track_points[(index-i)%len(self.track_points)] = None

        # Since we cannot modify a list that we are iterating, we set points to None
        # so we need to remove them now
        self.track_points = list(filter(None, self.track_points))

        # If debug is enabled, draw AI's new road points
        if self.debug:
            for point in self.track_points:
                point = (point[2] + Point2D(self.race.track_image.shape[0] // 2, self.race.track_image.shape[1] // 2)).as_integer_list()
                self.race.track_image[point] = [255, 0, 255]

        self.track_points_array = self.make_road_points_array(self.track_points)


if __name__ == '__main__':

    #import win32api
    import time

    reversed = True
    next_flip = np.random.randint(50, 100)

    # create objects
    race = Race(fps=10, debug=True)
    race.generate_track()
    car = RaceCar(speed=20, turn_radius=3)
    race.set_car(car, reversed=reversed)

    # For AI
    ai = AI(race, car, debug=True)

    frame = 0
    while True:

        # Manual steering
        '''
        throttle = 0
        steering = 0
        if win32api.GetAsyncKeyState(ord('A')):
            steering = -1
        elif win32api.GetAsyncKeyState(ord('D')):
            steering = 1
        if win32api.GetAsyncKeyState(ord('W')):
            throttle = 1
        elif win32api.GetAsyncKeyState(ord('S')):
            throttle = -1

        c.apply_steering(throttle, steering)
        '''
        # -- or --
        # For AI - AI steering
        action, new_lap = ai.step(reversed=reversed)

        # Step environment
        image = race.step()

        # Car is near to the edge
        if image is None:
            print('Out of track')
            exit()

        #time.sleep(1 / race.fps)

        # Add frame to the counter
        frame += 1

        # For AI - change driving offset randomly
        if not frame % 50:
            new_offset = np.random.randint(81) - 40
            ai.apply_offset(new_offset)

        # Change driving direction randomly
        if not frame % next_flip:
            reversed = not reversed
            next_flip = (np.random.randint(15, 50) if reversed else np.random.randint(50, 100))
