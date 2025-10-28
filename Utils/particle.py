import numpy as np
import random_numbers as rn

"""
This util file contains a particle class along with function for use in a particle filter.
The correct way to utilize the filter is in the following order

Initilize particle
estimate pose

Loop:
    move
    prediction_step
        (rejuvenation)????
    correction_step
    resampling_step
    estimate_pose
    rejuvenation_atep


"""



class Particle(object):
    """Data structure for storing particle information (state and weight)"""
    def __init__(self, x=0.0, y=0.0, theta=0.0, weight=0.0):
        self.x = x #x coordinate
        self.y = y #y coordinate
        self.theta = np.mod(theta, 2.0*np.pi) # wraps angle into raidans  [0 , 2\pi]
        self.weight = weight ## How important is this particle

    def getX(self):
        return self.x
        
    def getY(self):
        return self.y
        
    def getTheta(self):
        return self.theta
        
    def getWeight(self):
        return self.weight

    def setX(self, val):
        self.x = val

    def setY(self, val):
        self.y = val

    def setTheta(self, val):
        self.theta = np.mod(val, 2.0*np.pi)

    def setWeight(self, val):
        self.weight = val

def estimate_pose(particles_list):
    """Estimate the pose from particles by computing the average position and orientation over all particles. 
    This is not done using the particle weights, but just the sample distribution."""
    x_sum = 0.0
    y_sum = 0.0
    cos_sum = 0.0
    sin_sum = 0.0
     
    for particle in particles_list:
        x_sum += particle.getX()
        y_sum += particle.getY()
        cos_sum += np.cos(particle.getTheta())
        sin_sum += np.sin(particle.getTheta())
        
    flen = len(particles_list)
    if flen != 0:
        x = x_sum / flen
        y = y_sum / flen
        theta = np.arctan2(sin_sum/flen, cos_sum/flen)
    else:
        x = x_sum
        y = y_sum
        theta = 0.0
        
    return Particle(x, y, theta)
     
def move_particle(particle, delta_x, delta_y, delta_theta):
    """Move the particle by (delta_x, delta_y, delta_theta)
    Warning: we are assuming that delta_x and delta_y are given
    in world coordinates, this will not work if they are given in robot coordinates.
    """
    particle.x += delta_x
    particle.y += delta_y
    particle.theta = np.mod(particle.theta + delta_theta, 2.0 * np.pi)

def add_uncertainty(particles_list, sigma, sigma_theta):
    """Add some noise to each particle in the list. Sigma and sigma_theta is the noise
    variances for position and angle noise."""
    for particle in particles_list:
        particle.x += rn.randn(0.0, sigma)
        particle.y += rn.randn(0.0, sigma)
        particle.theta = np.mod(particle.theta + rn.randn(0.0, sigma_theta), 2.0 * np.pi) 


def add_uncertainty_von_mises(particles_list, sigma, theta_kappa):
    """Add some noise to each particle in the list. Sigma and theta_kappa is the noise
    variances for position and angle noise."""
    for particle in particles_list:
        particle.x += rn.randn(0.0, sigma)
        particle.y += rn.randn(0.0, sigma)
        particle.theta = np.mod(rn.rand_von_mises(particle.theta, theta_kappa), 2.0 * np.pi) - np.pi



def initialize_particles(num_particles):
    """ A simple function given to us to initialize a lot of particles
    """
    particles = []
    for i in range(num_particles):
        # Random starting points. 
        p = Particle(520.0*np.random.ranf() - 120.0, 420.0*np.random.ranf() - 120.0, np.mod(2.0*np.pi*np.random.ranf(), 2.0*np.pi), 1.0/num_particles)
        particles.append(p)

    return particles


#impiment motion model
#Used for the prediction step


# ------------------------ Self implimented particle filter---------------------------------------------

# PREDICTION STEP (SAMPLING STEP)  
# The motion modeL
def prediction_step(particles, distance, angle_change, sigma_d=2.0, sigma_theta=0.05):
    """
    Particle filter prediction step.
    1. Moves all particles deterministically based on odometry.
    2. Adds Gaussian uncertainty using add_uncertainty().
    
    Args:
        particles: List of Particle objects
        distance: Distance the robot drove in cm
        angle_change: Angle the robot turned in radians
        sigma_d: Position uncertainty (cm)
        sigma_theta: Angular uncertainty (radians)
    """
    # --- Step 1: Deterministic motion update ---
    for particle in particles:
        particle.theta = np.mod(particle.theta + angle_change, 2.0 * np.pi)
        particle.x += distance * np.cos(particle.theta)
        particle.y += distance * np.sin(particle.theta)
    
    # --- Step 2: Add Gaussian motion noise ---
    add_uncertainty(particles, sigma_d, sigma_theta)

#----------------------------------------------------------------------------------------------------

# CORRECTION STEP (WEIGHTING STEP)
def correction_step(particles, ids, dists, angles,LANDMARKS, sigma_d=10.0, sigma_theta=np.deg2rad(10)):
    """
    Update particle weights based on landmark observations. The code is based on algorithm 3 from
    https://medium.com/@mathiasmantelli/particle-filter-part-4-pseudocode-and-python-code-052a74236ba4
    It also needs to know which landmarks it should look for create something like
    
    Args:
        particles: List of Particle objects
        ids: ArUco IDs from camera (numpy array or None)
        dists: Distances in cm from camera (numpy array or None)
        angles: Angles in radians from camera (numpy array or None)
        LANDMARKS: dict mapping landmark id to coordinates, see example below
            LANDMARKS = {
                1: np.array([0.0, 0.0]),
                2: np.array([300.0, 0.0])}
        sigma_d: Distance measurement noise (cm)
        sigma_theta: Angle measurement noise (radians)
    """
    # No observations? Keep uniform weights
    if ids is None:
        for p in particles:
            p.weight = 1.0 / len(particles)
        return
    
    # Outer loop (Particle)
    # For each particle, compute weight
    for particle in particles:
        weight = 1.0
        
        #Inner Loop (Landmarks)
        # For each observed landmark
        for i in range(len(ids)):
            landmark_id = int(ids[i])
            
            if landmark_id not in LANDMARKS:
                continue  # Skip unknown landmarks
            
            landmark_pos = LANDMARKS[landmark_id]
            
            # -------------------------------
            # Distance Part (Formula 2)
            # -------------------------------

            #d^{(i)} = \sqrt{(l_x -x^{(i)})^2 + (l_y - y^{(i)})^2} essentially
            dx = landmark_pos[0] - particle.x #(l_x - x^(i))
            dy = landmark_pos[1] - particle.y #(l_y - y^(i))
            euclidean_distance = np.sqrt(dx**2 + dy**2) #euclidean distance

            #coefficient for Gaussian distance
            coef_distance = 1 / (np.sqrt(2 * np.pi * sigma_d**2))

            # Equation 2 from the pdf: distance probability
            prob_dist = coef_distance * np.exp(- (dists[i] - euclidean_distance)**2 / ( 2 * sigma_d**2)) 

            # -------------------------------
            # Orientation Part (Formula 3)
            # -------------------------------

            #phi(i)
            angle_world = np.arctan2(dy, dx)  # angle from particle to landmark in world frame
            phi_i = angle_world - particle.theta  # expected orientation measurement, relative to robot's current position
            phi_i = np.arctan2(np.sin(phi_i), np.cos(phi_i))  # wrap angle to [-pi, pi]

            #phi_M
            phi_M = angles[i]  # orientation measurement, relative to robot's current position

            # difference between angles (wrapped [-pi , pi])
            angle_diff = phi_M - phi_i
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff)) #wrap angle

            #coefficient for Gaussian distance
            coef_orientation = 1 / (np.sqrt(2 * np.pi * sigma_theta**2))
        
            #Prob dist using equation 3 from the pdf
            prob_orientation = coef_orientation * np.exp(- (angle_diff)**2 / ( 2 * sigma_theta**2))

            #calculate weight for this landmark observation
            weight *= prob_dist * prob_orientation

        
        particle.weight = weight
    
    # Normalize weights
    total = sum(p.weight for p in particles)
    if total > 0:
        for p in particles:
            p.weight /= total
    else:
        # All weights zero? Reset to uniform
        for p in particles:
            p.weight = 1.0 / len(particles)

#------------------------------------------------------
#The Resampling step
def resampling_step(particles):

    N = len(particles) # How many particles are we resampling
    if N == 0:     
        return []
    
    #Get weights
    w = np.array([p.weight for p in particles])  # FIXED: Added brackets
    w = w / np.sum(w)

    #Compute the cumulative distribution
    H = np.cumsum(w)

    # --- Resampling ---
    resampled_particles = []
    for _ in range(N):
        z = np.random.ranf()  # FIXED: Changed from rn.rand_uniform(0.0, 1.0) to np.random.ranf()
        i = np.searchsorted(H, z)
        selected = particles[i]
        # Copy selected particle into new set
        new_p = Particle(selected.x, selected.y, selected.theta, 1.0 / N)
        resampled_particles.append(new_p)

    return resampled_particles


#--------------------------------------------------------------------
# Rejuvenation step
def rejuvenation_step(particles, rejuvenation_ratio = 0.05, map_bounds= (520 , 420 , -120 ,- 120)):
    """
    introduces new particles into the filter to avoid colaps of diversity.
    
    args:
        particles: List of particles
        rejuvenation_ratio: fraction of particles to replace
        map_bounds:  (x_max, y_max , x_min, y_min)
    """
    N = len(particles)

    #Determine number of particles to rejuvenate
    n_random = int(N * rejuvenation_ratio)
    if n_random <= 0:
        return particles
    
    #extract map bounds
    x_max , y_max , x_min, y_min = map_bounds
    for i in range(n_random):
        particles[i] = Particle(
            np.random.uniform(x_min, x_max), # X
            np.random.uniform(y_min, y_max), # Y
            np.random.uniform(0, 2.0 * np.pi), # Theta
            1.0 /N #weight
        )

    return particles