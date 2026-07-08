# ABOUTME: QMC sequence generators (Roberts/Fibonacci/Korobov) + Voronoi-based posterior resampling.
# ABOUTME: Builds low-discrepancy latent lattices and Box-Muller / cell-sampling helpers.
import numpy as np
import torch
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist


def roberts_sequence(
        num_points,
        num_dims,
        root_iters=10_000,
    ):
    """
    Creates random numbers tiling a hybercube [0, 1]^d where d is `num_dims`.

    Code modified from:
    https://gist.github.com/carlosgmartin/1fd4e60bed526ec8ae076137ded6ebab
    """

    # Compute the unique positive root of f using the Newton-Raphson method.
    def f(x):
        return x ** (num_dims + 1) - x - 1

    def grad_f(x):
        return (num_dims + 1) * (x ** num_dims) - 1

    # Main loop.
    x = 1.0
    for i in range(root_iters):
        x = x - f(x) / grad_f(x)

    # Compute basis parameter
    basis = 1 - (1 / x ** (1 + torch.arange(0, num_dims)))

    # Return sequence without taking modulo 1
    return torch.arange(0, num_points)[:, None] * basis[None, :]

def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a+b
    return a

def gen_fib_basis(m):
    """
    Creates random numbers tiling a cube [0,1]^2 where m is element of the fibonacci sequence
    """

    n = fib(m)
    z = torch.tensor([1.,fib(m-1)])

    return torch.arange(0,n)[:,None]*z[None,:]/n


def gen_korobov_basis(a,
                      num_dims,
                      num_points):
    """
    Creates `num_points` random numbers tiling a cube [0,1]^d where d is `num_dims`

    some recommended values:
    num_points = 1021, a = 76
    num_points = 2039, a = 1487
    num_points = 4093, a = 1516
    see table 16.1 of owen for more
    these were constructed for num_dims \\in {8,12,24,32}
    this is a fibonacci lattice for num_dims = 2, a = Fib(m-1), n = Fib(m) for m >= 3
    """

    z = torch.tensor([a**k % num_points for k in range(num_dims)])
    base_pts = torch.arange(0,num_points)[:,None] * z[None,:]/num_points
    return base_pts

EPS = 1e-12
def box_muller(unif_2d_vars):

    """
    implements Box-Muller transform to turn two independent uniform random variables into two
    independent standard normal random variables. (See Art Owens Practical Monte Carlo eq. 4.9)

    oh and everything is in pytorch
    assumes 2d vars are in the shape B(atch size) x D(imension)
    """

    z1 = torch.sqrt(-2 * torch.log(unif_2d_vars[:,0]+EPS)) * torch.cos(2*torch.pi*unif_2d_vars[:,1])
    z2 = torch.sqrt(-2 * torch.log(unif_2d_vars[:,0]+EPS)) * torch.sin(2*torch.pi*unif_2d_vars[:,1])

    return torch.stack([z1,z2],dim=1)

######### For finer-grained sampling after the model is mostly fully trained on the original
######## grid. motivation here is that once the model has been heavily trained on the
####### first grid, most samples are wasted so we need to sample more finely for each data point

def get_default_voronoi(grid):
  """
  should take a grid and do everything needed to find a the voronoi cell around each grid point
  then, should center this at zero
  """
  (K,D) = grid.shape
  pts = grid.detach().cpu().numpy()
  k = np.argmin(np.linalg.norm(pts - 0.5*np.ones((D,)), axis=1))
  dists = cdist(pts[k][None,:],pts).ravel() # distance of all points to point closest to center of grid

  n_neighbors = min(50,len(pts)-1)
  partition =  np.argpartition(dists, n_neighbors)[:n_neighbors] # 20 nearest neighbors to central point. This should be enough points, but can increase to 50 just in case
  neighbors = np.setdiff1d(partition,[k]) # nearest neighbors not including center point

  vor = Voronoi(np.vstack([pts[k]] + [pts[kn] for kn in neighbors])) # voronoi partition of central point and all neighbors
  assert np.allclose(vor.points[0], pts[k])

  central_verts = vor.vertices[vor.regions[vor.point_region[0]]]

  zero_centered_cell = central_verts - pts[k]
  return torch.from_numpy(zero_centered_cell)

def sample_from_grid(grid,posterior,num_samples=1000):

    """
    takes as input a grid of voronoi cell centers,
    a weighting on those centers (based on some posterior),
    """

    #print(grid_len)
    #print(posterior.shape)
    #gen = np.random.default_rng()
    #fn = lambda posterior: gen.choice(grid_len,n_samples,replace=True,p=posterior)
    #print(fn(posterior.detach().cpu().numpy()).shape)
    #assert False
    #sample_fnc = np.vectorize(fn)
    #print(posterior.shape)
    (B1,N) = posterior.shape
    samples = torch.multinomial(posterior,num_samples,replacement=True).to(grid.device)
    (B2,K) = samples.shape
    assert B1 == B2
    #for ii,s in enumerate(samples):
    #    assert torch.all(s >= ii * N) and torch.all(s < (ii+1)*N)
    #samples = sample_fnc(posterior[None,:].detach().cpu().numpy()) #gen.choice(grid_len,n_samples,replace=True,p=posterior.detach().cpu().numpy()).squeeze()

    #assert False
    #posterior_weights = posterior[samples.view(np.prod]
    #print(posterior_weights.shape)
    #print(num_samples)
    #shifted_samples = samples + (torch.arange(0,B1)*N)[:,None]
    posterior_weights = [] #posterior.view(np.prod(posterior.shape))

    samples = samples.view(np.prod(samples.shape))
    #start = time.time()
    for ii in range(B1):
        posterior_weights.append(posterior[ii][samples])
    posterior_weights=torch.stack(posterior_weights,dim=0)
    if len(posterior_weights.shape) == 1:
        posterior_weights = posterior_weights[None,:]
    importance_weights = torch.mean(posterior_weights,axis=0,keepdims=True) # is this correct here? yes
    #end = time.time() - start
    #print(f"got weights in {end*1000:.2f}ms")
    #shifted_samples = shifted_samples.view(np.prod(samples.shape))


    #posterior_weights = posterior_weights[shifted_samples]
    #posterior_weights = posterior_weights.view(B1,K)
    assert (importance_weights.shape[0]) == 1 and (importance_weights.shape[1] == B2*K),print(importance_weights.shape)
    #print(posterior_weights.shape)
    #print(posterior_weights.shape)
    #print(grid.shape)
    #assert False
    #print(grid.shape)
    #print(samples.shape)
    #print(grid[samples].shape)
    return grid[samples],importance_weights

def sample_from_cells(cell_centers,cell_bounds):

    """
    cell_bounds should be kxd, where k is the number of bounding points, d is latent dim
    cell_centers should be bxd, where b is the number of centers per batch, and, d is the latent dim
    which means that shifted cells should end up as b x k x d
    """
    gen = np.random.default_rng()
    k = cell_bounds.shape[0]
    b = cell_centers.shape[0]
    dirichlet_weights = np.ones((k,))
    dirichlet_samples = torch.from_numpy(gen.dirichlet(dirichlet_weights,size=(b,1))).to(cell_centers.device)

    #print(cell_bounds.shape)
    #print(cell_centers.shape)
    #shifted_cells = cell_centers + cell_bounds[None,:,:]
    shifted_cells = cell_centers[:,None,:] + cell_bounds[None,:,:]
    #print(shifted_cells.shape)
    #assert False
    samples = torch.einsum('bkd,bjk->bjd',shifted_cells,dirichlet_samples).squeeze()

    samples[samples < 0] = 1 + samples[samples < 0]
    samples[samples >= 1] = samples[samples >=1] -1
    #print(samples.shape)
    return samples

def gen_samples_batch(grid,model,lp,samples,n_samples_total):

    n_per_s = n_samples_total // samples.shape[0]
    with torch.no_grad():
        posterior=model.posterior_probability(grid.to(model.device),samples.to(model.device),lp)

        centered_cell = get_default_voronoi(grid)

        sampled_shifts,importance_weights = sample_from_grid(grid,posterior,num_samples=n_per_s)

        samples = sample_from_cells(sampled_shifts,centered_cell)

    return samples.to(torch.float32),importance_weights.to(torch.float32)
