import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import interpolate
from torchdiffeq import odeint

activations = {'relu':    torch.nn.ReLU(),
               'sigmoid': torch.nn.Sigmoid(),
               'elu':     torch.nn.ELU(),
               'tanh':    torch.nn.Tanh(),
               'gelu':    torch.nn.GELU(),
               'silu':    torch.nn.SiLU(),
               'softplus': torch.nn.Softplus(beta=1, threshold=20),
               'leaky_relu': torch.nn.LeakyReLU()}

class NICE(torch.nn.Module):

    def __init__(self,params,
                 params_u,
                 hidden_num,
                 number_IC,
                 norm_params,
                 dim=2,
                 dtype=torch.float32):
        super(NICE, self).__init__()
        
        
        self.dtype = dtype
        self.hidden_num = hidden_num     
        self.dim = dim
        self.prm_e,self.prm_de,self.prm_ee,self.prm_dee,self.prm_s,self.prm_ds,self.prm_dt = norm_params
        self.solver = None
        self.NNf = self.constructor(params)
        self.NNu = self.constructor(params_u)
        self.relu = torch.nn.ReLU()
        self.e0 = torch.nn.Parameter(torch.zeros((number_IC,self.dim)),requires_grad=True)
        self.prm_u = np.linalg.norm(self.prm_s,axis=1)*np.linalg.norm(self.prm_e,axis=1)
        self.inference = None
        
    def constructor(self, params):
        '''
        Feed-forward artificial neural network constructor
        :params : [input layer # nodes, output layer # nodes, hidden layers # node, hidden activations]
        '''
        i_dim,o_dim,h_dim,act=params
        dim = i_dim
        layers = torch.nn.Sequential()
        for hdim in h_dim:
            layers.append(torch.nn.Linear(dim, hdim, dtype=self.dtype))
            layers.append(activations[act])
            dim = hdim
        layers.append(torch.nn.Linear(dim, o_dim, dtype=self.dtype))
        return layers
    
    def Normalize(self,inputs,prm):
        '''
        Normalize features
        :inputs : data
        :prm : normalization parameters
        '''
        return torch.divide(torch.add(inputs, -prm[1]), prm[0])
    
    def DeNormalize(self,outputs,prm):
        '''
        Denormalize features
        :output : dimensionless data
        :prm : normalization parameters
        '''
        return torch.add(torch.multiply(outputs, prm[0]), prm[1])
    
    def forward(self, t, y):
        uel = y[:,:self.dim]
        nel = self.Normalize(uel,self.prm_ee)
        if self.inference==False:
            if t>1.: ueps_dot = self.eps_dot[-1]  
            else: ueps_dot = self.eps_dot[int(t/self.prm_dt)]  
        else:
            ueps_dot = torch.zeros((len(self.idx),2))
            for i in range(len(self.idx)):
                ueps_dot[i,0] = torch.from_numpy(self.interp_dotev[self.idx[i]](t.detach().numpy()))
                ueps_dot[i,1] = torch.from_numpy(self.interp_dotes[self.idx[i]](t.detach().numpy()))
        neps_dot = self.Normalize(ueps_dot,self.prm_de).detach()
        nodes = self.NNf(torch.cat((neps_dot,nel),-1))
        node_el = nodes[:,:self.dim]
        uode_el = self.DeNormalize(node_el,self.prm_dee)     
        return uode_el
    
    def stress(self,uel):
        nel = self.Normalize(uel,self.prm_ee)  
        nu = self.NNu(nel)
        u = self.DeNormalize(nu,self.prm_u)
        ustress = (torch.autograd.grad(u,uel,grad_outputs=torch.ones_like(u),
                                     retain_graph=True,
                                     create_graph=True)[0])
        return ustress
    
    def integrate(self,u,y0,t,idx):
        self.eps_dot = u
        self.idx = idx
        y_ = odeint(self,y0,t,method=self.solver,options={"step_size": self.step_size})
        uel = y_[:,:,:self.dim]
        nel = self.Normalize(uel,self.prm_ee)  
        stress = self.stress(uel)
        nstress = self.Normalize(stress,self.prm_s)
        neps_dot = self.Normalize(self.eps_dot,self.prm_de)
        nodes = self.NNf(torch.cat((neps_dot,nel),-1))
        node_el = nodes[:,:,:self.dim]
        uode_el = self.DeNormalize(node_el,self.prm_dee)  
        uode_pl = self.eps_dot-uode_el
        sijdepij = torch.einsum('ijk,ijk->ij',stress[1:],uode_pl[:-1])
        return y_,nstress,sijdepij

    def init_interp(self,args,t):
        self.x = np.arange(args.shape[1])
        self.interp_dotev = []
        self.interp_dotes = []
        for i in range(len(self.x)):
            f = interpolate.interp1d(t, args[:,i,0],fill_value="extrapolate",kind="previous")
            g = interpolate.interp1d(t, args[:,i,1],fill_value="extrapolate",kind="previous")
            self.interp_dotev.append(f)
            self.interp_dotes.append(g)
            
            
class NICE2(torch.nn.Module):

    def __init__(self,params,
                 params_u,
                 hidden_num,
                 number_IC,
                 norm_params,
                 dim=2,
                 dtype=torch.float32):
        super(NICE, self).__init__()
        
        
        self.dtype = dtype
        self.hidden_num = hidden_num     
        self.dim = dim
        self.prm_e,self.prm_de,self.prm_ee,self.prm_dee,self.prm_s,self.prm_ds,self.prm_dt = norm_params
        self.solver = None
        self.NNf = self.constructor(params)
        self.NNu = self.constructor(params_u)
        self.relu = torch.nn.ReLU()
        self.e0 = torch.nn.Parameter(torch.zeros((number_IC,self.dim)),requires_grad=True)
        self.prm_u = np.linalg.norm(self.prm_s,axis=1)*np.linalg.norm(self.prm_e,axis=1)
        self.inference = None
        
    def constructor(self, params):
        '''
        Feed-forward artificial neural network constructor
        :params : [input layer # nodes, output layer # nodes, hidden layers # node, hidden activations]
        '''
        i_dim,o_dim,h_dim,act=params
        dim = i_dim
        layers = torch.nn.Sequential()
        for hdim in h_dim:
            layers.append(torch.nn.Linear(dim, hdim, dtype=self.dtype))
            layers.append(activations[act])
            dim = hdim
        layers.append(torch.nn.Linear(dim, o_dim, dtype=self.dtype))
        return layers
    
    def Normalize(self,inputs,prm):
        '''
        Normalize features
        :inputs : data
        :prm : normalization parameters
        '''
        return torch.divide(torch.add(inputs, -prm[1]), prm[0])
    
    def DeNormalize(self,outputs,prm):
        '''
        Denormalize features
        :output : dimensionless data
        :prm : normalization parameters
        '''
        return torch.add(torch.multiply(outputs, prm[0]), prm[1])
    
    def forward(self, t, y):
        uel = y[:,:self.dim]
        nel = self.Normalize(uel,self.prm_ee)
        if self.inference==False:
            if t>1.: ueps_dot = self.eps_dot[-1]  
            else: ueps_dot = self.eps_dot[int(t/self.prm_dt)]  
        else:
            ueps_dot = torch.zeros((len(self.idx),2))
            for i in range(len(self.idx)):
                ueps_dot[i,0] = torch.from_numpy(self.interp_dotev[self.idx[i]](t.detach().numpy()))
                ueps_dot[i,1] = torch.from_numpy(self.interp_dotes[self.idx[i]](t.detach().numpy()))
        neps_dot = self.Normalize(ueps_dot,self.prm_de).detach()
        nodes = self.NNf(torch.cat((neps_dot,nel),-1))
        node_el = nodes[:,:self.dim]
        uode_el = self.DeNormalize(node_el,self.prm_dee)     
        return uode_el
    
    def stress(self,uel):
        nel = self.Normalize(uel,self.prm_ee)  
        nu = self.NNu(nel)
        u = self.DeNormalize(nu,self.prm_u)
        ustress = (torch.autograd.grad(u,uel,grad_outputs=torch.ones_like(u),
                                     retain_graph=True,
                                     create_graph=True)[0])
        return ustress
    
    def integrate(self,u,y0,t,idx):
        self.eps_dot = u
        self.idx = idx
        y_ = odeint(self,y0,t,method=self.solver,options={"step_size": self.step_size})
        uel = y_[:,:,:self.dim]
        nel = self.Normalize(uel,self.prm_ee)  
        stress = self.stress(uel)
        nstress = self.Normalize(stress,self.prm_s)
        neps_dot = self.Normalize(self.eps_dot,self.prm_de)
        nodes = self.NNf(torch.cat((neps_dot,nel),-1))
        node_el = nodes[:,:,:self.dim]
        uode_el = self.DeNormalize(node_el,self.prm_dee)  
        uode_pl = self.eps_dot-uode_el
        sijdepij = torch.einsum('ijk,ijk->ij',stress[1:],uode_pl[:-1])
        return y_,nstress,sijdepij

    def init_interp(self,args,t):
        self.x = np.arange(args.shape[1])
        self.interp_dotev = []
        self.interp_dotes = []
        for i in range(len(self.x)):
            f = interpolate.interp1d(t, args[:,i,0],fill_value="extrapolate",kind="previous")
            g = interpolate.interp1d(t, args[:,i,1],fill_value="extrapolate",kind="previous")
            self.interp_dotev.append(f)
            self.interp_dotes.append(g)
            
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
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
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
