import pyparticleest.simulator as simulator
import pyparticleest.models.nlg as nlg
import numpy as np
import matplotlib.pyplot as plt

#state = tipx, tipy

class WhiskerFilter(nlg.NonlinearGaussianInitialGaussian):
    def __init__(self, x0, P0, Q, R, dt):
        super(WhiskerFilter, self).__init__(
                x0=x0, Px0=P0, Q=Q, R=R,
        )

        
    def calc_f(self, particles, u, t):
        tipx, tipy = particles[:, 0], particles[:, 1]

        return tipx, tipy

    def calc_g(self, particles, t):
        tipx, tipy = particles[:, 0], particles[:, 1]

        return tipx, tipy

# class Turner(object):
#     """ Class to test Kalman ability to distinguish objects """
#     def __init__(self, theta_max, period, phi=0, t = 0, dt = 0.15 ):
#         self.theta_max, self.period, self.phi, self.t, self.dt  = theta_max, period, phi, t, dt

#     def turn(self):
#         theta = self.theta_max * np.sin((2 * np.pi * self.t / self.period) + self.phi)
#         self.t += self.dt
#         return theta


# t_max = 100
# dt = 0.15

# turner = Turner(np.pi, 10, phi=0, dt=0.15)
# turner_angles = []


# theta0 = turner.theta_max * np.sin(turner.phi)
# omega0 = (2 * np.pi * turner.theta_max / turner.period) * np.cos(turner.phi)
# x0 = (theta0, omega0, 10)


# for i in range(t_max):
#     turner_angles.append(turner.turn())

# turner_angles = np.array(turner_angles)
# z = np.array([np.cos(turner_angles), np.sin(turner_angles)]).T

# # x0 = (0, np.pi/8, 1)
# P0 = np.array([
#     [0.1, 0, 0],
#     [0, 0.1, 0],
#     [0, 0,   6],
# ])

# Q = np.array([
#     [1e-9, 0,    0],
#     [0, 1e-9,    0],
#     [0, 0,     10],
# ])
# R = np.eye(2) * 100

# N = 100
# M = 10




# model = WhiskerFilter(x0, P0, Q, R, dt)

# sim = simulator.Simulator(model, u=None, y=z)
# sim.simulate(N, M, filter='PF', smoother='rsas', meas_first=True)

# (est_filt, w_filt) = sim.get_filtered_estimates()
# mean_filt = sim.get_filtered_mean()

# # print turner_angles
# # print mean_filt[:, 1]



# plt.plot(range(t_max), turner_angles, 'bo')
# plt.plot(range(t_max), mean_filt[:, 0], 'r--')

# plt.show()



# # class Tracker(object):
# #   def __init__(self, frame, regions, nRegions, particlesPerObj):
# #     self.particlesPerObj = particlesPerObj
# #     self.nTrajectories = nRegions

# #     trajectories = np.zeros((1, nRegions))
# #     for i in range(nTrajectories):
# #       particleFilter = particleFilter()
# #       initParticles()

# #       trajectories[i]

# #   def mergeTrack():
# #     d = 0


        



# # model = MultiTargetFilter(z0, P0, A)
# # sim = simulator.Simulator(model, u=None, )


# # import numpy as np
# # from numpy.random import uniform, randn, choice
# # import scipy.stats

# # class Partition(object):
# #   def __init__(self):
# #     self.particles = []
# #     self.weights = []

# # class Particles(object):
# #   def __init__(self):
# #     self.state = []

# # class Particle(object):
# #   def __init__(self):
# #     self.partitions = []



# # def ip_method(partition_max):
# #   for t in range(1, partition_max + 1):
# #     particles = partition.particles
# #     particles = predict(particles, A, Q)
# #     weights = update(particles, weights, z, R)


# #     new_particles = np.copy(particles)
# #     bias = np.zeros(new_particles.size)

# #     for i in range(len(new_particles)):
# #       j = choice(np.arange(particles.size), p=weights)
# #       new_particles[i, :] = particles[j, :]
# #       bias = weights[j]

# # def cp_method(partition_max, proposal_count):
# #   for t in range(1, partition_max + 1):
# #     for particle in particles:
# #       for m in proposals:
# #         particles = predict(particles, A, Q)
# #         weights = update(particles, weights, z, R)















# # def create_uniform_particles(n):
# #   particles = np.zeros((3, n))

# #   particles[0, :] = uniform(200, 350, size=n) #L

# #   particles[1, :] = uniform(-np.pi/2, np.pi/2, size=n) #theta
# #   particles[1, :] %= 2 * np.pi

# #   particles[2, :] = uniform(0, 2 * np.pi, size=n) #omega
# #   particles[2, :] %= 2 * np.pi

# #   # particles[:, 3] = uniform(0, 10, size=n) #period

# #   return particles


# # def predict(particles, A, Q, dt=1.):
# #   n_statevars, n_particles = particles.shape
# #   L, theta, omega = particles[0, :], particles[1, :], particles[2, :]

# #   particles = np.dot(A, particles) + (randn(n_statevars, n_particles)) * Q

# #   return particles


# # def update(particles, weights, z, R):
# #   L, theta, omega = particles[0, :], particles[1, :], particles[2, :]
# #   x, y = L * np.cos(theta), L * np.sin(theta)

# #   prediction = np.array([x, y]).T

# #   print weights
# #   weights.fill(1.)

# #   for i, coordinates in enumerate(z):
# #     weights *= scipy.stats.multivariate_normal.pdf(prediction, z[i], R)

# #   weights += 1.e-300
# #   weights /= sum(weights)

# #   return weights

# # def estimate(particles, weights):
# #   state = particles[:, 1:3]
# #   mean = np.average(state, weights=weights, axis=0)
# #   var = np.average((state - mean)**2, weights=weights, axis=0)

# #   return mean, var


# # def run_pf(n, iters=10, sensor_std_err=0.1, initial_x=None):
# #   particles = create_uniform_particles(n)
# #   Q = np.array([
# #     [4, np.pi / 6, np.pi / 4]
# #   ]).T

# #   R = np.eye(2)

# #   weights = np.zeros(n)
# #   dt = 1.
# #   T = 20
# #   A = np.array([
# #       [1, 0, 0],
# #       [0, 1, dt],
# #       [0, ((2 * np.pi / T) ** 2) * -dt, 1]
# #     ])

# #   for i in range(iters):
# #     z = [np.array([220, 230]) for i in range(7)]
# #     particles = predict(particles, A, Q)
# #     weights = update(particles, weights, z, R)

# #     print particles

# # run_pf(20)
# # # particles = initalize_particles(10)
# # # print update(particles)





