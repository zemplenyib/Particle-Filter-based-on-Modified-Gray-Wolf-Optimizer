import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import seaborn as sns
from glob import glob
from numpy.random import uniform
from numpy.random import randn
from numpy.random import random
from display_images import display_image
from display_images import display_scatter
from display_images import get_rectangle
from iou import iou
from filterpy.monte_carlo import systematic_resample
from filterpy.monte_carlo import residual_resample
from filterpy.monte_carlo import stratified_resample

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

def create_gaussian_particles(mean, std, N, dim):
  particles = np.empty((N, dim))
  for i in range(dim):
    particles[:, i] = mean[i] + (randn(N) * std[i])
  return particles

def create_clone_particles(initial,N,dim):
  dim = initial.size
  particles = np.tile(initial,(N,1))
  return particles

def get_histogram(frame, particle, w_init, h_init, visualize = False):
  if particle is not None:
    # Crop roi from image
    x1,x2,y1,y2 = get_rectangle(particle,w_init,h_init)
    roi = frame[y1:y2,x1:x2]
  else:
    roi = frame

  # Calculate histogram
  hist = cv2.calcHist([roi], [0, 1, 2], None, [8,8,8], ranges)
  hist = hist.flatten() / (np.sum(hist.flatten()) + epsilon)
  if visualize:
    fig, ax = plt.subplots()
    plt.imshow(roi)
    plt.show()
  return hist

def predict(particles, std, G = None, Q = None):
  if(G is not None and Q is not None):
    rng = np.random.default_rng()
    for i,particle in enumerate(particles):
      mean = G.dot(particle)
      particles[i] = rng.multivariate_normal(mean, Q)
  else:
    # No motion model, only dispersion
    rng = np.random.default_rng()
    particles = rng.normal(particles, std)
  return particles

def update(frame, particles, weights, target_hist, w_init, h_init, normalize = 'True'):
  for i,particle in enumerate(particles):
    particle_hist = get_histogram(frame,particle, w_init, h_init, visualize = False)
    weights[i] = 1-cv2.compareHist(target_hist, particle_hist, cv2.HISTCMP_BHATTACHARYYA) #cv2.HISTCMP_INTERSECT) #cv2.HISTCMP_CHISQR)
    
  if normalize:
    weights = weights / (np.sum(weights)+epsilon)
  return weights

def estimate(particles, weights, groundTruth, w_init, h_init, frame, target_hist):
  # State estimation by average of particles
  state_avg = np.sum(particles * weights[:,None], axis=0)

  # State estimation by largest weight
  state_lw = particles[np.argmax(weights)]

  # print ('Average:        ', state_avg)
  # print ('Highest weight: ', state_lw)
  # print ('True state:     ', tst[1,:])

  # Bounding box of the ground truth
  bb_gt = {'x1':groundTruth[0], 'y1':groundTruth[1],'x2':groundTruth[0]+groundTruth[2], 'y2':groundTruth[1]+groundTruth[3]}
  # Bounding box of the solution using the average
  # print('state_avg:' + str(state_avg))
  x1,x2,y1,y2 = get_rectangle(state_avg,w_init,h_init)

  if x1 < x2:
    bb_avg = {'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2}
  elif x1 > x2:
    bb_avg = {'x1':x2, 'y1':y1, 'x2':x1, 'y2':y2}
  else:
    bb_avg = {'x1':x1, 'y1':y1, 'x2':x2+1, 'y2':y2}

  if not y1 < y2:
    print(state_avg)
    print(x1,y1,x2-x1,y2-y1)
    print(groundTruth)
  # Bounding box of the solution using the largest weight
  # bb_lw  = {'x1':state_lw[0]-state_lw[2], 'y1':state_lw[1]-state_lw[3], 'x2':state_lw[0]+state_lw[2], 'y2':state_lw[1]+state_lw[3]}

  # Compute Intersection over Union to evaluate the solution. Higher values are 
  # better. IoU is bounded in [0,1]. In object detection an IoU >= 0.5 is usually 
  # considered a correct detection.
  IOU_avg = iou(bb_gt, bb_avg)
  # print ('IOU avg  = {:.3f}'.format(IOU_avg))
  # print ('IOU best = {:.3f}'.format(iou(bb_tst, bb_lw)))
  return state_avg, IOU_avg

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

def MGWO(N, dim, frame, target_hist, w_init, h_init, particles, weights, sigma, max_iter = 10):
  # print('MGWO0')
  # print(particles)
  # display_image(frame, w_init, h_init, 'MGWO0', size=1.0, particles = particles, weights = weights)
  
  # In case of losing object
  if not sum(weights)>0:
    height,width,layers = frame.shape
    particles = create_gaussian_particles([int(width/2), 1, int(height/2), 1, 1, 1], [100,5,100,5,5,1], len(particles), len(particles[0]))
    weights = update(frame, particles, weights, target_hist, w_init, h_init)
    print('Object lost')
    # display_image(frame, w_init, h_init, 'Object Lost', size=1.0, particles = particles, weights = weights)

  a = 2
  particles_new = np.empty((N, dim))
  weights_new = np.empty((N, 1))

  for t in range (max_iter):
    r1 = 0.5 + randn(N,3,dim)*sigma
    r2 = 0.5 + randn(N,3,dim)*sigma
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
      D_alpha = np.absolute(C[i,0,:]*X_alpha - particle)
      D_beta = np.absolute(C[i,1,:]*X_beta - particle)
      D_delta = np.absolute(C[i,2,:]*X_delta - particle)

      X_1 = X_alpha - A[i,0,:]*D_alpha
      X_2 = X_beta - A[i,1,:]*D_beta
      X_3 = X_delta - A[i,2,:]*D_delta

      particles_new[i] = (X_1 + X_2 + X_3)/3
    
    # Update particle if new solution is better
    weights     = update(frame, particles, weights, target_hist, w_init, h_init, normalize = False)
    weights_new = update(frame, particles_new, weights_new, target_hist, w_init, h_init, normalize = False)
    for i,(weight, weight_new) in enumerate(zip(weights, weights_new)):
      if weight_new > weight:
        particles[i] = particles_new[i]
        weights[i] = weights_new[i]
    # Update 'a' parameter
    a = 2 - 2*(np.sin(np.pi*t/max_iter/2))**2

    # Display particles
    # display_image(frame, w_init, h_init, 'MGWO_'+str(t+1), size=1.0, particles = particles, weights = weights)

  weights = weights / (np.sum(weights)+epsilon)
  return particles, weights

def import_data(dataset):
  folder = 'Datasets/' + dataset + '/' + dataset

  # Read images  
  images,filenames = load_images_from_folder(folder + '/img')

  # Read ground truth
  with open(folder + '/groundtruth_rect.txt', 'r') as f:
    gt = [[int(x) for x in line.split()] for line in f]
  gt = np.array(gt)
  return images, filenames, gt

# Export video
def export_video(frames_rgb, estimation, dataset, w_init, h_init, iou):
  iou_old = 0
  for filename in os.listdir():
    if filename[0:len(dataset)] == dataset:
      iou_old = float(filename[len(dataset)+1:len(dataset)+6])
  if float(iou) > iou_old:
    height,width,layers = frames_rgb[0].shape
    size = (width,height)
    out = cv2.VideoWriter(dataset+'_'+iou+'.avi',cv2.VideoWriter_fourcc(*'DIVX'),15,size)
    for frame_rgb,particle in zip(frames_rgb,estimation):
      x1,x2,y1,y2 = get_rectangle(particle,w_init,h_init)
      cv2.rectangle(frame_rgb,(x1, y1),(x2, y2),(255,0,0),2)
      out.write(cv2.cvtColor(frame_rgb,cv2.COLOR_RGB2BGR))
    out.release()
    if iou_old != 0:
      os.remove(dataset + '_' + '{:.3f}'.format(iou_old) + '.avi')

def run_pf(N, dataset, sigma, velocity, T, mgwo):
  # Import images, ground truth
  images, filenames, gt = import_data(dataset)
  initial_state = np.array([gt[0,0]+gt[0,2]/2,velocity[0],gt[0,1]+gt[0,3]/2,velocity[1], 0, 1])
  w_init = gt[0,2]
  h_init = gt[0,3]
  dim = initial_state.shape[0]

  # Create motion model matrices
  G = np.array([[1,T,0,0,0,0],[0,1,0,0,0,0],[0,0,1,T,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
  Qx = np.array([[1/4*T**4, 1/2*T**3],[1/2*T**3,T**2]])*sigma[0]**2
  Qy = np.array([[1/4*T**4, 1/2*T**3],[1/2*T**3,T**2]])*sigma[1]**2
  Q = np.zeros((6,6))
  Q[0:2,0:2] = Qx
  Q[2:4,2:4] = Qy
  Q[4,4]     = sigma[2]**2
  Q[5,5]     = sigma[3]**2

  
  # Create particles and weights
  if initial_state is not None:
    # Create the same particles N times
    particles = create_clone_particles(initial_state, N, dim)
  weights = np.ones(N) / N

  estimation = np.empty((len(images),dim))
  frames_rgb = []
  IOU_avg = []

  for index,frame_bgr in enumerate(images):
    if frame_bgr is not None:
      # Convert to rgb for display
      frame_rgb = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
      frames_rgb.append(frame_rgb)
      # Convert to lab for histogram
      frame_lab = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2Lab)

      # Create reference
      if index == 0:
        target_hist = get_histogram(frame_rgb, initial_state, w_init, h_init)
        # display_image(frame_rgb, w_init, h_init, '', size=1.0, particles = particles, weights = weights)

      # Move particles
      particles = predict(particles,sigma,G,Q)
      # display_image(frame_rgb, w_init, h_init, 'predict', size=1.0, particles = particles, weights = weights)

      # Evaluate particles and calculate weights
      frame_rgb = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
      weights = update(frame_rgb, particles, weights, target_hist, w_init, h_init)

      # Apply Modified Gray Wolf Optimizer
      if mgwo:
        frame_rgb = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
        particles, weights = MGWO(N = N, dim = dim, frame = frame_rgb, target_hist = target_hist, w_init = w_init, h_init = h_init, particles = particles, weights = weights, sigma = sigma[4], max_iter = mgwo_max_iter)
        # display_image(frame_rgb, w_init, h_init, 'MGWO', size=1.0, particles = particles, weights = weights)

      # Estimate current state
      state_estimate, IOU_avg_act = estimate(particles, weights, gt[index,:], w_init, h_init, frame_rgb, target_hist)
      IOU_avg.append(IOU_avg_act)
      estimation[index,:] = state_estimate      

      particles = resample(particles, weights, N)

      if index % 1 == 0:
        frame_rgb = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
        # display_image(frame_rgb, w_init, h_init, 'resample', size=1.0, particles = particles, weights = weights)
        frame_rgb = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
        # display_image(frame_rgb, w_init, h_init, 'estimate', size=1.0, particles = state_estimate, weights = weights)
      index += 1

  # Export video with estimation
  # export_video(frames_rgb, estimation, dataset, w_init, h_init, '{:.3f}'.format(sum(IOU_avg)/len(IOU_avg)))
  print('Average IOU = {:.3f}'.format(sum(IOU_avg)/len(IOU_avg)))
  # for index,frame_rgb in enumerate(frames_rgb):
    # display_image(frame_rgb, w_init, h_init, 'IOU avg  = {:.3f}'.format(IOU_avg[index]), size=1.0, particles = estimation[index,:])
  return sum(IOU_avg)/len(IOU_avg)


# Constants
ranges = [0, 256, 0, 256, 0, 256]
epsilon = 0.000001
mgwo_max_iter = 10

dataset = 'BlurBody'
sigma_x = 1.73 #1.4 #6 #1.4
sigma_y = 0.64 #1.4 #6 #1.4
sigma_theta = 1.96 #2.5
sigma_s = 0.023 #0.025
sigma_mgwo = 0.0458 #0.1 #0.05 #0.1
velocity = [1,1] #[1,1]
mgwo = True
T = 1

run_pf(100, dataset, sigma = [sigma_x,sigma_y,sigma_theta,sigma_s, sigma_mgwo], velocity = velocity, T = T, mgwo = mgwo)
