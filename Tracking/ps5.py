"""Problem Set 5: Object Tracking and Pedestrian Detection"""

import cv2
import numpy as np

from ps5_utils import run_kalman_filter, run_particle_filter

# I/O directories
input_dir = "input"
output_dir = "output"



# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        # Updated initial state estimate
        self.state = np.array([init_x, init_y, 0., 0.])
        
        # for MOT
        self.width = 0
        self.height = 0
        self.age = 0
        
        # Initial error covariance matrix
        self.P = np.eye(4)
        
        # Process noise covariance
        self.Q = Q
        
        # Measurement noise covariance
        self.R = R

        # State transition matrix
        self.A = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        
        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

    def predict(self):

        # Predict the next state
        self.state = np.dot(self.A, self.state)

        # Predict the error covariance
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def correct(self, meas_x, meas_y):

        # Compute Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(np.dot(np.dot(self.H, self.P), self.H.T) + self.R))

        # Update state estimate
        z = np.array([meas_x, meas_y])
        self.state = self.state + np.dot(K, (z - np.dot(self.H, self.state)))

        # Update error covariance
        self.P = np.dot((np.eye(4) - np.dot(K, self.H)), self.P)

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]

    
class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles', 10000)
        self.sigma_exp = kwargs.get('sigma_exp', 1.0)
        self.sigma_dyn = kwargs.get('sigma_dyn', 1.0)
        self.template_rect = kwargs.get('template_coords')
        self.template = template
        
        self.frame = frame

        # init particles
        self.particles = np.random.rand(self.num_particles, 2)
        width = self.frame.shape[1]
        height = self.frame.shape[0]
        self.particles = self.particles * np.array((width, height))

        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        return np.sum(np.abs(template.astype(float) - frame_cutout.astype(float)))
        
        

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.

        Returns:
            numpy.array: particles data structure.
        """
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        return self.particles

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        # For each particle:
        #  - Add random velocity to particle (input)
        #  - Ensure particles do not go out of frame
        #  - Calculate error metric
        # Update weights
        # Resample particles

        for i in range(self.num_particles):
            u = np.random.normal(0, self.sigma_dyn)
            v = np.random.normal(0, self.sigma_dyn)
            self.particles[i][0] += u
            self.particles[i][1] += v

            # limiting particles to stay within the frame
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]
            particle_min_x = self.template_rect['w']//2
            particle_min_y = self.template_rect['h']//2
            particle_max_x = frame_width - self.template_rect['w']//2 - 1
            particle_max_y = frame_height - self.template_rect['h']//2 - 1
            self.particles[i][0] = max(particle_min_x, min(particle_max_x, self.particles[i][0]))
            self.particles[i][1] = max(particle_min_y, min(particle_max_y, self.particles[i][1]))

            frame_cutout = None
            if self.template.shape[0] % 2:
                frame_cutout = frame[int(self.particles[i][1]) - self.template.shape[0]//2:int(self.particles[i][1]) + self.template.shape[0]//2+1,
                                     int(self.particles[i][0]) - self.template.shape[1]//2:int(self.particles[i][0]) + self.template.shape[1]//2]
            if self.template.shape[1] % 2:
                frame_cutout = frame[int(self.particles[i][1]) - self.template.shape[0]//2:int(self.particles[i][1]) + self.template.shape[0]//2,
                                     int(self.particles[i][0]) - self.template.shape[1]//2:int(self.particles[i][0]) + self.template.shape[1]//2+1]
            
            if self.template.shape[0] % 2 and self.template.shape[1] % 2:
                frame_cutout = frame[int(self.particles[i][1]) - self.template.shape[0]//2:int(self.particles[i][1]) + self.template.shape[0]//2+1,
                                     int(self.particles[i][0]) - self.template.shape[1]//2:int(self.particles[i][0]) + self.template.shape[1]//2+1]
            else:
                frame_cutout = frame[int(self.particles[i][1]) - self.template.shape[0]//2:int(self.particles[i][1]) + self.template.shape[0]//2,
                                     int(self.particles[i][0]) - self.template.shape[1]//2:int(self.particles[i][0]) + self.template.shape[1]//2]
            self.weights[i] = self.get_error_metric(self.template, frame_cutout)

        
        self.weights = np.max(self.weights) - self.weights
        self.weights = self.weights**16
        self.weights /= np.sum(self.weights)
        self.resample_particles()
    
    def render(self, frame_in):
        """Renders the current state of the particle filter on the frame.

        Draws particles and template rectangle on the frame.

        Returns:
            numpy.array: Frame with particles and template rectangle drawn.
        """
        for particle in self.particles:
            cv2.circle(frame_in, (int(particle[0]), int(particle[1])), 1, (255, 255, 255), 2)

        x_mean = np.mean(self.particles[:,0]).astype(int)
        y_mean = np.mean(self.particles[:,1]).astype(int)
        cv2.rectangle(frame_in, (x_mean - self.template_rect['w']//2, y_mean - self.template_rect['h']//2), (x_mean + self.template_rect['w']//2, y_mean + self.template_rect['h']//2), (255, 255, 255), 2)
        

class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha_init = kwargs.get('alpha', 0.)  # required by the autograder
        self.alpha = self.alpha_init
    
    def check_convergence(self, particles, threshold):
        mean_position = np.mean(particles, axis=0)
        distances = np.linalg.norm(particles - mean_position, axis=1)
        dispersion = np.mean(distances)
        return dispersion < threshold

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        best_particle_index = np.argmax(self.weights)
        best_particle = self.particles[best_particle_index]
        
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        particle_min_x = self.template_rect['w']//2
        particle_min_y = self.template_rect['h']//2
        particle_max_x = frame_width - self.template_rect['w']//2 - 1
        particle_max_y = frame_height - self.template_rect['h']//2 - 1
        best_particle[0] = max(particle_min_x, min(particle_max_x, best_particle[0]))
        best_particle[1] = max(particle_min_y, min(particle_max_y, best_particle[1]))

        best_template = None
        if self.template.shape[0] % 2:
            best_template = frame[int(best_particle[1]) - self.template.shape[0]//2:int(best_particle[1]) + self.template.shape[0]//2+1,
                                 int(best_particle[0]) - self.template.shape[1]//2:int(best_particle[0]) + self.template.shape[1]//2]
        if self.template.shape[1] % 2:
            best_template = frame[int(best_particle[1]) - self.template.shape[0]//2:int(best_particle[1]) + self.template.shape[0]//2,
                                 int(best_particle[0]) - self.template.shape[1]//2:int(best_particle[0]) + self.template.shape[1]//2+1]
        
        if self.template.shape[0] % 2 and self.template.shape[1] % 2:
            best_template = frame[int(best_particle[1]) - self.template.shape[0]//2:int(best_particle[1]) + self.template.shape[0]//2+1,
                                 int(best_particle[0]) - self.template.shape[1]//2:int(best_particle[0]) + self.template.shape[1]//2+1]
        else:
            best_template = frame[int(best_particle[1]) - self.template.shape[0]//2:int(best_particle[1]) + self.template.shape[0]//2,
                                 int(best_particle[0]) - self.template.shape[1]//2:int(best_particle[0]) + self.template.shape[1]//2]
        

        # if self.template is not grey scale, convert it to grey scale
        if len(self.template.shape) > 2:
            self.template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        
        # normalize the current template and the best template
        current_template = cv2.normalize(self.template.astype(np.float32), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        best_template = cv2.normalize(best_template.astype(np.float32), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)


        new_template = cv2.normalize((best_template * self.alpha) + (current_template * (1-self.alpha)), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        self.template = new_template.astype(np.uint8)

        # if the particles converged into small area, using high alpha will make the template to be updated quickly
        if self.check_convergence(self.particles, 10) and self.alpha_init != 0.:
            print("Converged")
            self.alpha = 0.08
        else:
            print("Not converged")
            self.alpha = self.alpha_init
        
        # Call the base class method to perform regular particle filtering
        super(AppearanceModelPF, self).process(frame)


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.sigma_scale = kwargs.get('sigma_scale')

        # creating particles with 3 dimensions (x, y, scale)
        self.particles = np.random.rand(self.num_particles, 3)
        width = self.frame.shape[1]
        height = self.frame.shape[0]
        self.particles = self.particles * np.array((width, height, 1.0))

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """

        for i in range(self.num_particles):
            u = np.random.normal(0, self.sigma_dyn)
            v = np.random.normal(0, self.sigma_dyn)
            s = np.random.normal(0.5, self.sigma_scale)
            self.particles[i][0] += u
            self.particles[i][1] += v
            self.particles[i][2] += s

            # limiting particles to stay within the frame
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]
            particle_min_x = self.template_rect['w']//2
            particle_min_y = self.template_rect['h']//2
            particle_min_scale = 0.5
            particle_max_x = frame_width - self.template_rect['w']//2 - 1
            particle_max_y = frame_height - self.template_rect['h']//2 - 1
            particle_max_scale = 1.5
            self.particles[i][0] = max(particle_min_x, min(particle_max_x, self.particles[i][0]))
            self.particles[i][1] = max(particle_min_y, min(particle_max_y, self.particles[i][1]))
            self.particles[i][2] = max(particle_min_scale, min(particle_max_scale, self.particles[i][2]))

            frame_cutout = None
            if self.template.shape[0] % 2:
                frame_cutout = frame[int(self.particles[i][1]) - self.template.shape[0]//2:int(self.particles[i][1]) + self.template.shape[0]//2+1,
                                     int(self.particles[i][0]) - self.template.shape[1]//2:int(self.particles[i][0]) + self.template.shape[1]//2]
                frame_cutout = cv2.resize(frame_cutout, (0,0), fx=self.particles[i][2], fy=self.particles[i][2])
            if self.template.shape[1] % 2:
                frame_cutout = frame[int(self.particles[i][1]) - self.template.shape[0]//2:int(self.particles[i][1]) + self.template.shape[0]//2,
                                     int(self.particles[i][0]) - self.template.shape[1]//2:int(self.particles[i][0]) + self.template.shape[1]//2+1]
                frame_cutout = cv2.resize(frame_cutout, (0,0), fx=self.particles[i][2], fy=self.particles[i][2])
            if self.template.shape[0] % 2 and self.template.shape[1] % 2:
                frame_cutout = frame[int(self.particles[i][1]) - self.template.shape[0]//2:int(self.particles[i][1]) + self.template.shape[0]//2+1,
                                     int(self.particles[i][0]) - self.template.shape[1]//2:int(self.particles[i][0]) + self.template.shape[1]//2+1]
                frame_cutout = cv2.resize(frame_cutout, (0,0), fx=self.particles[i][2], fy=self.particles[i][2])
            else:
                frame_cutout = frame[int(self.particles[i][1]) - self.template.shape[0]//2:int(self.particles[i][1]) + self.template.shape[0]//2,
                                     int(self.particles[i][0]) - self.template.shape[1]//2:int(self.particles[i][0]) + self.template.shape[1]//2]
                frame_cutout = cv2.resize(frame_cutout, (0,0), fx=self.particles[i][2], fy=self.particles[i][2])
            
            resized_template = cv2.resize(self.template, (0,0), fx=self.particles[i][2], fy=self.particles[i][2])
            frame_cutout = cv2.resize(frame_cutout, (resized_template.shape[1], resized_template.shape[0]))
            self.weights[i] = self.get_error_metric(resized_template[:frame_cutout.shape[0], :frame_cutout.shape[1]], frame_cutout)

        
        self.weights = np.max(self.weights) - self.weights
        self.weights = self.weights**16
        self.weights /= np.sum(self.weights)
        self.resample_particles()

    def render(self, frame_in):
        """Renders the current state of the particle filter on the frame.

        Draws particles and template rectangle on the frame.

        Returns:
            numpy.array: Frame with particles and template rectangle drawn.
        """
        for particle in self.particles:
            cv2.circle(frame_in, (int(particle[0]), int(particle[1])), 1, (255, 255, 255), 2)


        best_particle_index = np.argmax(self.weights)
        best_particle = self.particles[best_particle_index]
        

        x_mean = np.mean(self.particles[:,0]).astype(int)
        y_mean = np.mean(self.particles[:,1]).astype(int)
        best_particle_scale = best_particle[2]
        cv2.rectangle(frame_in, (x_mean - int((self.template_rect['w']//2)), y_mean - int((self.template_rect['h']//2))), (x_mean + int((self.template_rect['w']//2)), y_mean + int((self.template_rect['h']//2))), (255, 255, 255), 2)



def part_1b(obj_class, template_loc, save_frames, input_folder):
    Q = 0.1 * np.eye(4)  # Process noise array
    R = 0.1 * np.eye(2)  # Measurement noise array
    NOISE_2 = {'x': 7.5, 'y': 7.5}
    out = run_kalman_filter(obj_class, input_folder, NOISE_2, "matching",
                            save_frames, template_loc, Q, R)
    return out


def part_1c(obj_class, template_loc, save_frames, input_folder):
    Q = 0.1 * np.eye(4)  # Process noise array
    R = 0.1 * np.eye(2)  # Measurement noise array
    NOISE_1 = {'x': 2.5, 'y': 2.5}
    out = run_kalman_filter(obj_class, input_folder, NOISE_1, "hog",
                            save_frames, template_loc, Q, R)
    return out


def part_2a(obj_class, template_loc, save_frames, input_folder):
    num_particles = 100  # Define the number of particles
    sigma_mse = 10.  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10.  # Define the value of sigma for the particles movement (dynamics)

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_2b(obj_class, template_loc, save_frames, input_folder):
    num_particles = 1000  # Define the number of particles
    sigma_mse = 10.  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10.  # Define the value of sigma for the particles movement (dynamics)

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_3(obj_class, template_rect, save_frames, input_folder):
    num_particles = 400  # Define the number of particles
    sigma_mse = 10.  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 20.  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.001  # Set a value for alpha
    

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        # input video
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        alpha=alpha,
        template_coords=template_rect)  # Add more if you need to
    return out


def part_4(obj_class, template_rect, save_frames, input_folder):
    num_particles = 400  # Define the number of particles
    sigma_md = 10.  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 1.  # Define the value of sigma for the particles movement (dynamics)
    sigma_scale = 10. # Define the value of sigma for the scale change


    out = run_particle_filter(
        obj_class,
        input_folder,
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_md,
        sigma_dyn=sigma_dyn,
        sigma_scale=sigma_scale,
        template_coords=template_rect)  # Add more if you need to
    return out

def part_6(obj_class, template_rect, save_frames, input_folder):
    num_particles = 400  # Define the number of particles
    sigma_md = 10.  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10.  # Define the value of sigma for the particles movement (dynamics)
    sigma_scale = 10. # Define the value of sigma for the scale change

    out = run_particle_filter(
        obj_class,
        input_folder,
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_md,
        sigma_dyn=sigma_dyn,
        sigma_scale=sigma_scale,
        template_coords=template_rect)  # Add more if you need to
    return out
