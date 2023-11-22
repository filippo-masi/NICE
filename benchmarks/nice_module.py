import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import interpolate
from torchdiffeq import odeint

# Dictionary of activation functions
activations = {'relu': torch.nn.ReLU(),
               'sigmoid': torch.nn.Sigmoid(),
               'elu': torch.nn.ELU(),
               'tanh': torch.nn.Tanh(),
               'gelu': torch.nn.GELU(),
               'silu': torch.nn.SiLU(),
               'softplus': torch.nn.Softplus(beta=1, threshold=20),
               'leaky_relu': torch.nn.LeakyReLU()}

class NICE(torch.nn.Module):
    '''
    Neural Integration for Constitutive Equations (NICE)
    '''

    def __init__(self, params_f, params_u, number_IC, norm_params, dim=2, dtype=torch.float32):
        super(NICE, self).__init__()

        # Set data type and dimension
        self.dtype = dtype
        self.dim = dim

        # Unpack normalization parameters
        self.prm_e, self.prm_de, self.prm_rho, self.prm_z, self.prm_dz, self.prm_s, self.prm_dt = norm_params

        # Calculate fractions for el and de normalization
        frac = 0.5
        self.prm_ee = self.prm_e * frac
        self.prm_dee = self.prm_de * frac

        # Initialize solver and neural networks for evolution and energy
        self.solver = None
        self.NeuralNetEvolution = self.constructor(params_f)
        self.NeuralNetEnergy = self.constructor(params_u)
        self.relu = torch.nn.ReLU()

        # Initialize elastic strain parameter and normalization factor
        self.e0 = torch.nn.Parameter(torch.zeros((number_IC, self.dim)), requires_grad=True)
        self.prm_u = np.linalg.norm(self.prm_s, axis=1) * np.linalg.norm(self.prm_e, axis=1)

        # Initialize inference parameter
        self.inference = None

    def constructor(self, params):
        '''
        Feed-forward artificial neural network constructor
        :params : [input layer # nodes, output layer # nodes, hidden layers # node, hidden activations]
        '''
        i_dim, o_dim, h_dim, act = params
        dim = i_dim
        layers = torch.nn.Sequential()
        for hdim in h_dim:
            # Add hidden layer and activation function
            layers.append(torch.nn.Linear(dim, hdim, dtype=self.dtype))
            layers.append(activations[act])
            dim = hdim
        # Add output layer
        layers.append(torch.nn.Linear(dim, o_dim, dtype=self.dtype))
        return layers

    def Normalize(self, inputs, prm):
        '''
        Normalize features
        :inputs : data
        :prm : normalization parameters
        '''
        return torch.divide(torch.add(inputs, -prm[1]), prm[0])

    def DeNormalize(self, outputs, prm):
        '''
        Denormalize features
        :output : dimensionless data
        :prm : normalization parameters
        '''
        return torch.add(torch.multiply(outputs, prm[0]), prm[1])

    def forward(self, t, y):
        # Extract elastic strain and normalize
        uel = y[:, :self.dim]
        nel = self.Normalize(uel, self.prm_ee)

        # Determine total strain rate (ueps_dot)
        if self.inference == False:
            if t > 1.:
                ueps_dot = self.eps_dot[-1]
            else:
                ueps_dot = self.eps_dot[int(t / self.prm_dt)]
        else:
            # Interpolate external data for inference
            ueps_dot = torch.zeros((len(self.idx), 2))
            for i in range(len(self.idx)):
                ueps_dot[i, 0] = torch.from_numpy(self.interp_dotev[self.idx[i]](t.detach().numpy()))
                ueps_dot[i, 1] = torch.from_numpy(self.interp_dotes[self.idx[i]](t.detach().numpy()))
        neps_dot = self.Normalize(ueps_dot, self.prm_de).detach()

        # Extract mass density and dissipative state variable, and normalize
        urho = y[:, self.dim:self.dim + 1]
        uz = y[:, self.dim + 1:]
        nrho = self.Normalize(urho, self.prm_rho)
        nz = self.Normalize(uz, self.prm_z)

        # Feed-forward neural network for evolution equations
        nodes = self.NeuralNetEvolution(torch.cat((neps_dot, nel, nrho, nz), -1))
        node_pl = nodes[:, :self.dim]
        node_z = nodes[:, self.dim:self.dim + 1]

        # Calculate state variable evolution
        uode_rho = urho * ueps_dot[:, :1]
        uode_el = ueps_dot - self.DeNormalize(node_pl, self.prm_de)
        uode_z = self.DeNormalize(node_z, self.prm_dz)

        # Concatenate results
        return torch.cat((uode_el, uode_rho, uode_z), -1)

    def stress(self, X, grads=False):
        # Extract inputs: elastic strain, mass density, dissipative state variable
        uel, urho, uz = X

        # Normalize inputs
        nel = self.Normalize(uel, self.prm_ee)
        nrho = self.Normalize(urho, self.prm_rho)
        nz = self.Normalize(uz, self.prm_z)

        # Concatenate normalized inputs
        svars = torch.cat((nel, nrho, nz), -1)

        # Neural network for energy
        nu = self.NeuralNetEnergy(svars)

        # De-normalize energy
        u = self.DeNormalize(nu, self.prm_u)

        # Calculate stress and chemical potential
        ustress = (torch.autograd.grad(u,
                                       uel,
                                       grad_outputs=torch.ones_like(u),
                                       retain_graph=True,
                                       create_graph=True)[0])
        umu = (torch.autograd.grad(u,
                                   urho,
                                   grad_outputs=torch.ones_like(u),
                                   retain_graph=True,
                                   create_graph=True)[0])

        # Calculate thermodynamic pressure
        pT = torch.cat((umu * urho - u, torch.zeros(u.shape)),-1)
        ustress += pT

        if grads == True:
            # If gradients are requested, calculate the gradient with respect to dissipative state variable
            utau = -(torch.autograd.grad(u,
                                         uz,
                                         grad_outputs=torch.ones_like(u),
                                         retain_graph=True,
                                         create_graph=True)[0])
            return ustress, utau
        else:
            return ustress

    def integrate(self, u, y0, t, idx):
        # Integrate the ODE using torchdiffeq
        self.eps_dot = u
        self.idx = idx
        y_ = odeint(self, y0, t, method=self.solver, options={"step_size": self.step_size})

        # Extract normalized variables
        uel = y_[:, :, :self.dim]
        nel = self.Normalize(uel, self.prm_ee)
        urho = y_[:, :, self.dim:self.dim + 1]
        uz = y_[:, :, self.dim + 1:]
        nrho = self.Normalize(urho, self.prm_rho)
        nz = self.Normalize(uz, self.prm_z)

        # Calculate stress and dissipative forces
        stress, tau = self.stress([uel, urho, uz], grads=True)
        X = torch.cat((nel, nrho, nz), -1)

        # Calculate evolution equations
        neps_dot = self.Normalize(self.eps_dot, self.prm_de)
        nodes = self.NeuralNetEvolution(torch.cat((neps_dot, X), -1))
        node_pl = nodes[:, :, :self.dim]
        uode_el = self.eps_dot - self.DeNormalize(node_pl, self.prm_de)
        uode_pl = self.DeNormalize(node_pl, self.prm_de)

        # Calculate dissipated rate
        d = torch.einsum('ijk,ijk->ij', stress[1:], uode_pl[:-1])

        # Calculate dissipative rate due to the dissipative state variable
        node_z = nodes[:, :, self.dim:self.dim + 1]
        uode_z = self.DeNormalize(node_z, self.prm_dz)
        d += torch.einsum('ijk,ijk->ij', tau[1:], uode_z[:-1])

        return y_, stress, d

    def init_interp(self, args, t):
        # Initialize interpolation for external data
        self.x = np.arange(args.shape[1])
        self.interp_dotev = []
        self.interp_dotes = []
        for i in range(len(self.x)):
            f = interpolate.interp1d(t, args[:, i, 0], fill_value="extrapolate", kind="previous")
            g = interpolate.interp1d(t, args[:, i, 1], fill_value="extrapolate", kind="previous")
            self.interp_dotev.append(f)
            self.interp_dotes.append(g)

    def find_elastic_strain(self, eps_e, state):
        # Find elastic strain using a root-finding method
        rho, z, sigma = state
        eps_e_tensor = torch.from_numpy(eps_e.reshape(-1, 2))
        eps_e_tensor.requires_grad = True
        ueps_e_tensor = self.DeNormalize(eps_e_tensor, self.prm_ee)
        ustress = self.stress([ueps_e_tensor, rho, z])
        rhs = self.Normalize(sigma, self.prm_s).detach() - self.Normalize(ustress, self.prm_s).detach().numpy()
        return rhs.reshape(-1)


class NICE_reduced(torch.nn.Module):
    '''
    Neural Integration for Constitutive Equations (NICE) - Reduced Version
    '''

    def __init__(self, params_evolution, params_energy, number_IC, norm_params, dim=2, dtype=torch.float32):
        super(NICE_reduced, self).__init__()

        # Set data type and dimension
        self.dtype = dtype
        self.dim = dim

        # Unpack normalization parameters
        self.prm_e, self.prm_de, self.prm_s, self.prm_dt = norm_params

        # Calculate fractions for el and de normalization
        frac = 0.5
        self.prm_ee = self.prm_e * frac
        self.prm_dee = self.prm_de * frac

        # Initialize solver and neural networks for evolution and energy
        self.solver = None
        self.NeuralNetEvolution = self.constructor(params_evolution)
        self.NeuralNetEnergy = self.constructor(params_energy)
        self.relu = torch.nn.ReLU()

        # Initialize elastic strain parameter and normalization factor
        self.e0 = torch.nn.Parameter(torch.zeros((number_IC, self.dim)), requires_grad=True)
        self.prm_u = np.linalg.norm(self.prm_s, axis=1) * np.linalg.norm(self.prm_e, axis=1)
        self.inference = None

    def constructor(self, params):
        '''
        Feed-forward artificial neural network constructor
        :params : [input layer # nodes, output layer # nodes, hidden layers # node, hidden activations]
        '''
        i_dim, o_dim, h_dim, act = params
        dim = i_dim
        layers = torch.nn.Sequential()
        for hdim in h_dim:
            layers.append(torch.nn.Linear(dim, hdim, dtype=self.dtype))
            layers.append(activations[act])
            dim = hdim
        layers.append(torch.nn.Linear(dim, o_dim, dtype=self.dtype))
        return layers

    def Normalize(self, inputs, prm):
        '''
        Normalize features
        :inputs : data
        :prm : normalization parameters
        '''
        return torch.divide(torch.add(inputs, -prm[1]), prm[0])

    def DeNormalize(self, outputs, prm):
        '''
        Denormalize features
        :output : dimensionless data
        :prm : normalization parameters
        '''
        return torch.add(torch.multiply(outputs, prm[0]), prm[1])

    def forward(self, t, y):
        # Extract elastic strain and normalize
        uel = y[:, :self.dim]
        nel = self.Normalize(uel, self.prm_ee)

        # Determine total strain rate (ueps_dot)
        if self.inference == False:
            if t > 1.:
                ueps_dot = self.eps_dot[-1]
            else:
                ueps_dot = self.eps_dot[int(t / self.prm_dt)]
        else:
            # Interpolate external data for inference
            ueps_dot = torch.zeros((len(self.idx), 2))
            for i in range(len(self.idx)):
                ueps_dot[i, 0] = torch.from_numpy(self.interp_dotev[self.idx[i]](t.detach().numpy()))
                ueps_dot[i, 1] = torch.from_numpy(self.interp_dotes[self.idx[i]](t.detach().numpy()))
        neps_dot = self.Normalize(ueps_dot, self.prm_de).detach()

        # Feed-forward neural network for evolution
        nodes = self.NeuralNetEvolution(torch.cat((neps_dot, nel), -1))
        node_el = nodes[:, :self.dim]

        # De-normalize the output
        uode_el = self.DeNormalize(node_el, self.prm_dee)

        return uode_el

    def stress(self, uel):
        # Normalize elastic strain
        nel = self.Normalize(uel, self.prm_ee)

        # Neural network for energy
        nu = self.NeuralNetEnergy(nel)

        # De-normalize energy
        u = self.DeNormalize(nu, self.prm_u)

        # Calculate stress
        ustress = (torch.autograd.grad(u, uel, grad_outputs=torch.ones_like(u),
                                      retain_graph=True, create_graph=True)[0])
        return ustress

    def integrate(self, u, y0, t, idx):
        # Integrate the ODE using torchdiffeq
        self.eps_dot = u
        self.idx = idx
        y_ = odeint(self, y0, t, method=self.solver, options={"step_size": self.step_size})

        # Extract normalized variables
        uel = y_[:, :, :self.dim]
        nel = self.Normalize(uel, self.prm_ee)

        # Calculate stress
        stress = self.stress(uel)
        nstress = self.Normalize(stress, self.prm_s)
        neps_dot = self.Normalize(self.eps_dot, self.prm_de)

        # Feed-forward neural network for evolution
        nodes = self.NeuralNetEvolution(torch.cat((neps_dot, nel), -1))
        node_el = nodes[:, :, :self.dim]

        # De-normalize the output
        uode_el = self.DeNormalize(node_el, self.prm_dee)

        # Calculate the dissipation rate
        uode_pl = self.eps_dot - uode_el
        sijdepij = torch.einsum('ijk,ijk->ij', stress[1:], uode_pl[:-1])

        return y_, stress, sijdepij

    def init_interp(self, args, t):
        # Initialize interpolation for external data
        self.x = np.arange(args.shape[1])
        self.interp_dotev = []
        self.interp_dotes = []
        for i in range(len(self.x)):
            f = interpolate.interp1d(t, args[:, i, 0], fill_value="extrapolate", kind="previous")
            g = interpolate.interp1d(t, args[:, i, 1], fill_value="extrapolate", kind="previous")
            self.interp_dotev.append(f)
            self.interp_dotes.append(g)

    def find_elastic_strain(self, eps_e, sigma):
        # Find elastic strain using a root-finding method
        eps_e_tensor = torch.from_numpy(eps_e.reshape(-1, 2))
        eps_e_tensor.requires_grad = True
        ueps_e_tensor = self.DeNormalize(eps_e_tensor, self.prm_ee)
        ustress = self.stress(ueps_e_tensor)
        rhs = self.Normalize(sigma, self.prm_s).detach() - self.Normalize(ustress, self.prm_s).detach().numpy()
        return rhs.reshape(-1)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after the last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): Trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    
    def __call__(self, val_loss, model):
        """
        Monitors the validation loss and performs early stopping if needed.

        Args:
            val_loss (float): Current validation loss.
            model: PyTorch model.

        Returns:
            None
        """
        score = val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            
        elif score > self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves the model when validation loss decreases.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def slice_data(x, ntrainval, ntest):
    """
    Slices the data into training and testing sets.

    Args:
        x: Input data.
        ntrainval (int): Number of samples for training and validation.
        ntest (int): Number of samples for testing.

    Returns:
        Tuple: Sliced training/validation data, sliced testing data.
    """
    return x[:, ntrainval], x[:, ntest]

def get_params(x, norm=False, vectorial_norm=False):
    '''
    Compute normalization parameters:
        - normalize ([-1,1]) component by component (vectorial_norm = True)
        - normalize ([-1,1]) (vectorial_norm = False, norm = True)
        - standardize (vectorial_norm = False, norm = False)

    Args:
        x: Input data.
        norm (bool): Normalize data to [-1,1].
        vectorial_norm (bool): Normalize data component by component (along axis = 1).

    Returns:
        torch.Tensor: Normalization parameters.
    '''
    if vectorial_norm == False:
        if norm == True:
            # Normalize to [-1, 1]
            A = 0.5 * (np.amax(x) - np.amin(x))
            B = 0.5 * (np.amax(x) + np.amin(x))
        else:
            # Standardize (mean = 0, std = 1)
            A = np.std(x, axis=(0, 1))
            B = np.mean(x, axis=(0, 1))
    else:
        # Normalize component by component (along axis = 1)
        dim = x.shape[-1]
        u_max = np.zeros((dim,))
        u_min = np.zeros((dim,))
        for i in np.arange(dim):
            u_max[i] = np.amax(x[:, i])
            u_min[i] = np.amin(x[:, i])
        A = (u_max - u_min) / 2.
        B = (u_max + u_min) / 2.
        A[A == 0] = 1
    return torch.tensor(np.array([np.float64(A), np.float64(B)]))