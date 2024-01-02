import math

def check_sh_degree(sh_features):
  assert len(sh_features.shape) == 3, f"SH features must have 3 dimensions, got {sh_features.shape}"

  n_sh = sh_features.shape[2]
  n = int(math.sqrt(n_sh))

  assert n * n == n_sh, f"SH feature count must be square, got {n_sh} ({sh_features.shape})"
  return (n - 1)

def num_sh_features(deg):
  return (deg + 1) ** 2 


sh0 = 0.282094791773878

def rgb_to_sh(rgb):
    return (rgb - 0.5) / sh0

def sh_to_rgb(sh):
    return sh * sh0 + 0.5