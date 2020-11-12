import torch
import math
import numpy as np
from torchdiffeq import odeint

class GRUODECell(torch.nn.Module):
    """predicts continuous function in between datapoints
    hidden states can up updated jointly"""
    def __init__(self, input_size: Union[int, None], hidden_size: int, 
                 bias: bool=True, impute: bool=False) -> None:
        """
        """
        super().__init__()
        self.input_size = input_size
        if hidden_size:
            self.hidden_size = hidden_size
        else:
            self.hidden_size = 2*input_size
        self.bias = bias
        self.impute = impute
        self.lin_xz = torch.nn.Linear(in_features=input_size, out_features=hidden_size,
                                      bias=bias)
        self.lin_xn = torch.nn.Linear(in_features=input_size, out_features=hidden_size,
                                      bias=bias)

        self.lin_hz = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size, 
                                      bias=False)
        self.lin_hn = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size, 
                                      bias=False)

    def forward(self, x, h, delta_t: int):
        """
        Returns a change due to one step of using GRU-ODE for all h.
        Args:
            x        input values
            h        hidden state (current)
            delta_t  time step size
        Returns:
            Updated h
        """
        if x:
            xz = self.lin_xz(x)
            xn = self.lin_xn(x)
        else:  # 'autonomous in original implementation
            xz = xn = torch.zeros_like(h)
        
        # from fullGRU in original implementation
        if impute:
        xz, xn    self.lin_x = torch.nn.Linear(input_size, hidden_size * 3, bias=bias)


        
        # z is update gate state
        z = torch.sigmoid(xz + self.lin_hz(h))
        # n aka g is the update vector of the GRU
        n = torch.tanh(xn + self.lin_hn(z * h))

        dh = (1 - z) * (n - h)
        return dh


class GRUBayesCell(torch.nn.Module):
    """updates (filters) based on new datapoint
    updated and loss only calculated on time series with an 
    observation at a given point in time"""
    pass


class GRUObservationCell(torch.nn.Module):
    """Implements discrete update based on the received observations."""

    def __init__(self, input_size: int, hidden_size: int, prep_hidden: int,
                 bias:bool=True, logvar:bool=False):
        super().__init__()
        self.gru_d     = torch.nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)
        self.gru_debug = torch.nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)

        ## prep layer and its initialization
        std            = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep    = torch.nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
        self.bias_prep = torch.nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))
        self.logvar = logvar
        self.input_size  = input_size
        self.prep_hidden = prep_hidden
        self.var_eps     = 1e-6

    def forward(self, h, p, X_obs, M_obs, i_obs) -> Tuple[torch.Tensor, float]:
        ## only updating rows that have observations
        p_obs = p[i_obs]
        mean, var = torch.chunk(p_obs, 2, dim=1)
        if self.logvar:
            logvar = var
            sigma = torch.exp(0.5 * logvar)
        else:
            ## making var non-negative and also non-zero (by adding a small value)
            adjusted_var = torch.abs(var) + self.var_eps
            sigma = torch.sqrt(adjusted_var)
        error = (X_obs - mean) / sigma

        if self.logvar:
            log_lik_c = np.log(np.sqrt(2*np.pi))
            loss = 0.5 * ((torch.pow(error, 2) + logvar + 2*log_lik_c) * M_obs)

        else:
            ## log normal loss, over all observations
            loss = 0.5 * ((torch.pow(error, 2) + torch.log(adjusted_var)) * M_obs).sum()


        ## TODO: try removing X_obs (they are included in error)
        gru_input = torch.stack([X_obs, mean, adjusted_var, error], dim=2).unsqueeze(2)
        gru_input = torch.matmul(gru_input, self.w_prep).squeeze(2) + self.bias_prep
        gru_input.relu_()
        ## gru_input is (sample x feature x prep_hidden)
        gru_input = gru_input.permute(2, 0, 1)
        gru_input = (gru_input * M_obs).permute(1, 2, 0).contiguous().view(-1, self.prep_hidden * self.input_size)

        temp = h.clone()
        temp[i_obs] = self.gru_d(gru_input, h[i_obs])
        h = temp

        return h, loss


def init_weights(m):
    """
    description:
    Args:
        m: 
    Returns:
    """
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.05)


class NNFOwithBayesianJumps(torch.nn.Module):
    ## Neural Negative Feedback ODE with Bayesian jumps
    pass

def compute_KL_loss(p_obs, X_obs, M_obs, obs_noise_std=1e-2, logvar:bool=True) -> float:
    """
    Args:
        p_obs: 
        X_obs:
        M_obs:
        obs_noise_std:
        logvar: 
    Returns:
        the KL divergence
    """
    def gaussian_KL(mu_1, mu_2, sigma_1, sigma_2):
        return (torch.log(sigma_2) - torch.log(sigma_1) 
        + (torch.pow(sigma_1,2)+torch.pow((mu_1 - mu_2),2)) / (2*sigma_2**2) - 0.5)

    obs_noise_std = torch.tensor(obs_noise_std)
    if logvar:
        mean, var = torch.chunk(p_obs, 2, dim=1)
        std = torch.exp(0.5*var)
    else:
        mean, var = torch.chunk(p_obs, 2, dim=1)
        ## making var non-negative and also non-zero (by adding a small value)
        std       = torch.pow(torch.abs(var) + 1e-5,0.5)

    return ((gaussian_KL(mu_1=mean, mu_2=X_obs, sigma_1=std, sigma_2=obs_noise_std)
            * M_obs)
            .sum()
            )

class GRUODEBayesSeq(torch.nn.Module):
    """On top of the architecture described in the main bulk of this paper, we 
    also propose a variant which process the sporadic inputs sequentially. In 
    other words, GRU-Bayes will update its prediction on the hidden h for one 
    input dimension after the other rather than jointly. We call this approach
    GRU-ODE-Bayes-seq."""
    def __init__(self, input_size, hidden_size, p_hidden, prep_hidden, bias=True,
                 cov_size=1, cov_hidden=1, classification_hidden=1, logvar=True,
                 mixing=1, dropout_rate=0, obs_noise_std=1e-2, full_gru_ode=False):
        super().__init__()
        self.obs_noise_std = obs_noise_std
        self.classification_model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, classification_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(classification_hidden, 1, bias=bias),
            )
        self.gru_c = GRUODECell(input_size=2*input_size, hidden_size=hidden_size,
                                bias=bias, impute=full_gru_ode)


class SeqGRUBayes(torch.nn.Module):
    """
    Inputs to forward:
        h      tensor of hiddens
        X_obs  PackedSequence of observation values
        F_obs  PackedSequence of feature ids
        i_obs  indices of h that have been observed
    Returns updated h.
    """