particles = np.empty((N, 4))
  particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
  particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
  particles[:, 2] = uniform(w_range[0], w_range[1], size=N)
  particles[:, 3] = uniform(h_range[0], h_range[1], size=N)
  return particles