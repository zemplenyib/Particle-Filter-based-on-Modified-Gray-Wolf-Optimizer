import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import seaborn as sns
from glob import glob
from numpy.random import uniform
from numpy.random import randn
from display_images import display_image
from display_images import display_scatter
from iou import iou
from filterpy.monte_carlo import systematic_resample
from filterpy.monte_carlo import residual_resample

# Constants
ranges = [0, 256, 0, 256, 0, 256]
epsilon = 0.000001
mgwo_max_iter = 10

def load_images_from_folder(folder):
  images = []
  filenames = []
  for filename in sorted(os.listdir(folder)):
      img = cv2.imread(os.path.join(folder,filename))
      if img is not None:
          images.append(img)
          filenames.append(filename)
  return images,filename

def create_uniform_particles(x_range, y_range, w_range, h_range, N):
  particles = np.empty((N, 4))
  particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
  particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
  particles[:, 2] = uniform(w_range[0], w_range[1], size=N)
  particles[:, 3] = uniform(h_range[0], h_range[1], size=N)
  return particles

def create_gaussian_particles(mean, std, N):
  particles = np.empty((N, 4))
  particles[:, 0] = mean[0] + (randn(N) * std[0])
  particles[:, 1] = mean[1] + (randn(N) * std[1])
  particles[:, 2] = mean[2] + (randn(N) * std[2])
  particles[:, 3] = mean[3] + (randn(N) * std[3])
  return particles

def create_clone_particles(initial,N,dim):
  dim = initial.size
  particles = np.tile(initial,(N,1))
  return particles

def get_histogram(frame, particle, w_init, h_init):
  # Crop roi from image
  x1,x2,y1,y2 = get_rectangle(particle,w_init,h_init)
  roi = frame[y1:y2,x1:x2]
  # Calculate histogram
  hist = cv2.calcHist([roi], [0, 1, 2], None, [8,8,8], ranges)
  hist = hist.flatten() / (np.sum(hist.flatten()) + epsilon)
  return hist

def get_rectangle(particle, w_init, h_init):
  s = particle[5]
  w = w_init*s
  h = h_init*s
  x1 = particle[0]-w/2
  x2 = particle[0]+w/2
  y1 = particle[2]-h/2
  y2 = particle[2]+h/2
  return x1,x2,y1,y2

def predict(particles, std):
  # No motion model, only dispersion
  rng = np.random.default_rng()
  particles = rng.normal(particles, std)
  particles = particles.astype(int)


  return particles

def update(frame, particles, weights, target_hist):
  for i,particle in enumerate(particles):
    particle_hist = get_histogram(frame,particle)
    weights[i] = cv2.compareHist(target_hist, particle_hist, cv2.HISTCMP_BHATTACHARYYA) #cv2.HISTCMP_INTERSECT) #cv2.HISTCMP_CHISQR) 
  weights = weights / np.sum(weights)
  return weights

def estimate(particles, weights, tst):
  # State estimation by average of particles
  state_avg = np.sum(particles * weights[:,None], axis=0).astype(int)

  # State estimation by largest weight
  state_lw = particles[np.argmax(weights)]

  # print ('Average:        ', state_avg)
  # print ('Highest weight: ', state_lw)
  # print ('True state:     ', tst[1,:])

  # Bounding box of the true state
  # bb_tst = {'x1':tst[1,0]-tst[1,2], 'y1':tst[1,1]-tst[1,3],'x2':tst[1,0]+tst[1,2], 'y2':tst[1,1]+tst[1,3]}
  # Bounding box of the solution using the average
  # bb_avg = {'x1':state_avg[0]-state_avg[2], 'y1':state_avg[1]-state_avg[3], 'x2':state_avg[0]+state_avg[2], 'y2':state_avg[1]+state_avg[3]}
  # Bounding box of the solution using the largest weight
  # bb_lw  = {'x1':state_lw[0]-state_lw[2], 'y1':state_lw[1]-state_lw[3], 'x2':state_lw[0]+state_lw[2], 'y2':state_lw[1]+state_lw[3]}

  # Compute Intersection over Union to evaluate the solution. Higher values are 
  # better. IoU is bounded in [0,1]. In object detection an IoU >= 0.5 is usually 
  # considered a correct detection.
  # print ('IOU avg  = {:.3f}'.format(iou(bb_tst, bb_avg)))
  # print ('IOU best = {:.3f}'.format(iou(bb_tst, bb_lw)))
  return state_avg

def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights.resize(len(particles))
    weights.fill (1.0 / len(weights))

def resample(particles, weights, N):
  indexes = systematic_resample(weights)
  resample_from_index(particles, weights, indexes)
  assert np.allclose(weights, 1/N)
  return particles

def update_target(frame, state_estimate, target_hist, lmbd):
  frame_hist = get_histogram(frame, state_estimate)
  target_hist = target_hist + lmbd * frame_hist
  target_hist =  target_hist.flatten() / np.sum(target_hist.flatten())
  return target_hist

def neff(weights):
  return 1. / np.sum(np.square(weights))

def MGWO(N, dim, frame, target_hist, particles = None, weights = np.array([5,1,4,2,3]), max_iter = 10):
  a = 2
  particles_new = np.empty((N, 4))
  weights_new = np.empty((N, 1))

  for t in range (max_iter):
    r1 = 0.5 + randn(3,4)*0.2
    r2 = 0.5 + randn(3,4)*0.2

    A = 2*a*r1-a
    C = 2*r2
    
    ind = np.argpartition(weights, -3)[-3:]
    ind = ind[np.argsort(weights[ind])]

    alpha = ind[2]
    beta = ind[1]
    delta = ind[0]

    X_alpha = particles[alpha]
    X_beta = particles[beta]
    X_delta = particles[delta]


    for i, particle in enumerate(particles):
      D_alpha = np.absolute(C[0,:]*X_alpha - particle)
      D_beta = np.absolute(C[1,:]*X_beta - particle)
      D_delta = np.absolute(C[2,:]*X_delta - particle)

      X_1 = X_alpha - A[0,:]*D_alpha
      X_2 = X_beta - A[1,:]*D_beta
      X_3 = X_delta - A[2,:]*D_delta

      particles_new[i] = (X_1 + X_2 + X_3)/3
    
    # Update particle if new solution is better
    particles_new = particles_new.astype(int)
    weights_new = update(frame, particles_new, weights_new, target_hist)
    for i,(weight, weight_new) in enumerate(zip(weights, weights_new)):
      if weight_new > weight:
        particles[i] = particles_new[i]
        weights[i] = weights_new[i]
    # Update 'a' parameter
    a = 2 - 2*(np.sin(np.pi*t/max_iter/2))**2

def import_data(dataset):
  folder = 'Datasets/' + dataset + '/' + dataset

  # Read images  
  images,filenames = load_images_from_folder(folder + '/img')

  # Read ground truth
  with open(folder + '/groundtruth_rect.txt', 'r') as f:
    gt = [[int(x) for x in line.split()] for line in f]
  gt = np.array(gt)

  return images, filenames, gt  

def run_pf(N, dataset, sigma, velocity):
  images, filenames, gt = import_data(dataset)
  initial_state = np.array([gt[0,0]+g[0,2]/2,velocity,gt[0,1]+g[0,3]/2,velocity, 0, 1])
  dim = initial_state.shape[1]
  
  # Create particles and weights
  if initial_state is not None:
    # Create the same particles N times
    particles = create_clone_particles(initial_state, N, dim)
  weights = np.ones(N) / N

  for index,frame_bgr in enumerate(images):
    if frame_bgr is not None:
      # Convert to rgb for display
      frame_rgb = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
      # Convert to lab for histogram
      frame_lab = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2Lab)

      # Create reference
      if index == 0:
        target_hist = get_histogram(frame_rgb, initial_state, w_init, h_init)
        display_image(frame_rgb, index, size=1.0, particles = particles, weights = weights)

      # Move particles
      particles = predict(particles,sigma)

      # Evaluate particles and calculate weights
      weights = update(frame_rgb, particles, weights, target_hist)

      # Apply Modified Gray Wolf Optimizer
      MGWO(N = N, dim = dim, frame = frame_rgb, target_hist = target_hist, particles = particles, weights = weights, max_iter = mgwo_max_iter)

      # Estimate current state
      tst = np.array([[650,345,299,63],[650,345,299,63]])
      state_estimate = estimate(particles, weights, tst)

      particles = resample(particles, weights, N)

      if index % 1 == 0:
        display_image(frame_rgb, index, size=1.0, particles = particles, weights = weights)

      index += 1


dataset = 'BlurBody'
sigma_x = 1
sigma_y = 1
sigma_theta = 1
sigma_s = 1
velocity = 1

# run_pf(100, dataset, sigma = [sigma_x,sigma_y,sigma_theta,sigma_s], velocity = velocity)
asd = np.array([[1,2,3],[4,5,6]])
print(asd.shape[0])



