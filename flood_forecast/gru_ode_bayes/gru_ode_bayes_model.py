import torch
import math
import numpy as np
from torchdiffeq import odeint
from typing import Optional, Tuple

class _GRUODECell(torch.nn.Module):
    """predicts continuous function in between datapoints
    hidden states can up updated jointly
    Args:
        input_size:
        hidden_size:
        bias: include bias in neurons
        impute: 
        minimal: implement 'minimal' version per appendix G
    Returns: instantiated cell

    Re imputation: In case continuous observations or control signals are 
    available, they can be naturally fed to the GRU-ODE input x(t). For example,
    in the case of clinical trials, the administered daily doses of the drug 
    under study can be used to define a continuous input signal. If no 
    continuous input is available, then nothing is fed as x(t) and the resulting
    ODE in Eq. 3 is autonomous, with g(t) and z(t) only depending on h(t).
    """
    def __init__(self, input_size: Optional[int], hidden_size: int, 
                 bias: bool=True, impute: bool=False, minimal:bool=False) -> None:
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

    def forward(self, x: Optional[torch.Tensor], hidden_state: torch.Tensor, 
                delta_t: int) -> torch.Tensor:
        """
        Returns a change due to one step of using GRU-ODE for all h.
        Args:
            x              input values. Optional, can calculate without new values
            hidden_state:  hidden state (current)
            delta_t:       time step size
        Returns:
            Updated hidden state
        """
        if self.impute: # ie non-autonomous
            xz = xn = torch.zeros_like(hidden_state)
        else:  # 'autonomous in original implementation
            xz = self.lin_xz(x)
            xn = self.lin_xn(x) 

        # z in original paper; updated gate state
        updated_gate = torch.sigmoid(xz + self.lin_hz(hidden_state))
        # g in paper; n in author's code
        update_vec = torch.tanh(xn + self.lin_hn(updated_gate * hidden_state))

        # Reset gate state: n in original code
        if self.minimal:
            reset_get = updated_gate
        else:
            reset_gate = torch.sigmoid(xr + self.lin_hr(hidden_state))
      
        return (1 - updated_gate) * (reset_gate - hidden_state)


class _GRUObservationCell(torch.nn.Module):
    """Implements discrete update based on the received observations.
    init Args:
        input_size:
        hidden_size:
        prep_hidden:
        bias: include bias term
        logvar: Use log variance, false-> use variance
    
    """

    def __init__(self, input_size: int, hidden_size: int, prep_hidden: int,
                 bias:bool=True, use_logvar:bool=False):
        super().__init__()
        self.gru_d = torch.nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)

        ## prep layer and its initialization
        std = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep = torch.nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
        self.bias_prep = torch.nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))
        self.use_logvar = use_logvar
        self.input_size  = input_size
        self.prep_hidden = prep_hidden
        self.var_eps = 1e-6

    def forward(self, h, p, X_obs, M_obs, i_obs) -> Tuple[torch.Tensor, float]:
        ## only updating rows that have observations
        p_obs = p[i_obs]
        mean, var = torch.chunk(p_obs, 2, dim=1)
        if self.use_logvar:
            logvar = var
            sigma = torch.exp(0.5 * logvar)
        else:
            ## making var non-negative and also non-zero (by adding a small value)
            adjusted_var = torch.abs(var) + self.var_eps
            sigma = torch.sqrt(adjusted_var)
        error = (X_obs - mean) / sigma

        if self.use_logvar:
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
        gru_input = ((gru_input * M_obs)
                     .permute(1, 2, 0)
                     .contiguous()
                     .view(-1, self.prep_hidden * self.input_size))

        temp = h.clone()
        temp[i_obs] = self.gru_d(gru_input, h[i_obs])
        h = temp

        return h, loss


def init_weights(m: torch.nn) -> None:
    """
    description:
    Args:
        m: the torch model
    Returns:
        None. Has side effect of initializing model weights for model m.
    """
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.05)


class NNFOwithBayesianJumps(torch.nn.Module):
    """Neural Negative Feedback ODE with Bayesian jumps
    The smoother variable computes the classification loss as a weighted average of the
    projection of the latents at each observation. impute feeds the parameters of the 
    distribution to GRU-ODE at each step.
    
    Args:
        input_size: 
        hidden_size:
        p_hidden:
        prep_hidden:
        bias:
        cov_size:
        cov_hidden:
        classification_hidden:
        use_logvar:
        mixing: mixing hyperparameter for prejump_loss and postjump_loss. 
                loss = prejump_loss + mixing*postjump_loss
        dropout_rate
        minimal: use minimal version of GRU-ODE cell
        solver: ["euler", "midpoint", "dopri5]
        impute:
    Returns:

    """
    def __init__(self, input_size, hidden_size, p_hidden, prep_hidden, bias=True, 
                 cov_size=1, cov_hidden=1, classification_hidden=1, 
                 use_logvar=True, 
                 mixing=1, dropout_rate:float=0, minimal:Bool=False, 
                 solver:str="euler", impute = True, store_hist: bool=False):
        super().__init__()

        self.impute = impute
        self.use_logvar = use_logvar
        self.minimal = minimal
        self.p_model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, p_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(p_hidden, 2 * input_size, bias=bias),
        )

        self.classification_model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, classification_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(classification_hidden, 1, bias=bias)
        )
        self.gru_c = _GRUODECell(input_size=2*input_size, hidden_size=hidden_size,
                                bias=bias, impute=full_gru_ode)
        self.gru_obs = _GRUObservationCell(input_size, hidden_size, prep_hidden, 
                                          bias=bias, use_logvar=self.use_logvar)
        
        self.covariates_map = torch.nn.Sequential(
            torch.nn.Linear(cov_size, cov_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(cov_hidden, hidden_size, bias=bias),
            torch.nn.Tanh()
        )
        assert solver in ["euler", "midpoint", "dopri5"], "Solver must be either 'euler' or 'midpoint' or 'dopri5'."

        self.solver = solver
        self.store_hist = store_hist
        self.input_size = input_size
        self.use_logvar = use_logvar
        self.mixing = mixing 
        self.apply(init_weights)

    def ode_step(self, h, p, delta_t, current_time):
        """Executes a single ODE step."""
        eval_times = torch.tensor([0],device = h.device, dtype = torch.float64)
        eval_ps = torch.tensor([0],device = h.device, dtype = torch.float32)
        if self.impute is False:
            p = torch.zeros_like(p)
            
        if self.solver == "euler":
            h = h + delta_t * self.gru_c(p, h)
            p = self.p_model(h)

        elif self.solver == "midpoint":
            k  = h + delta_t / 2 * self.gru_c(p, h)
            pk = self.p_model(k)

            h = h + delta_t * self.gru_c(pk, k)
            p = self.p_model(h)

        elif self.solver == "dopri5":
            assert self.impute==False #Dopri5 solver is only compatible with autonomous ODE.
            solution, eval_times, eval_vals = odeint(self.gru_c, h, 
                torch.tensor([0,delta_t]), method=self.solver, 
                options={"store_hist":self.store_hist})
            if self.store_hist:
                eval_ps = self.p_model(torch.stack([ev[0] for ev in eval_vals]))
            eval_times = torch.stack(eval_times) + current_time
            h = solution[1,:,:]
            p = self.p_model(h)
        
        current_time += delta_t
        return h, p, current_time, eval_times, eval_ps
    
def forward(self, times: np.array, time_ptr, X: torch.Tensor, 
            M: torch.Tensor, obs_idx, delta_t: float, T, cov,
            return_path=False, smoother = False, class_criterion = None, 
            labels=None) -> list:
        """
        Args:
            times         vetor of observation times
            time_ptr      start indices of data for a given time
            X             data tensor
            M             mask tensor (1.0 if observed, 0.0 if unobserved)
            obs_idx       observed patients of each datapoint (indexed within the current minibatch)
            delta_t       time step for Euler
            T             total time
            cov           static covariates for learning the first h0
            return_path   whether to return the path of h
        Returns:
            h             hidden state at final time (T)
            loss          loss of the Gaussian observations
        """

        h = self.covariates_map(cov)
        p = self.p_model(h)
        current_time = 0.0
        counter = 0

        prejump_loss = 0 # frmly loss_1
        postjump_loss = 0 # KL between p_updated and the actual sample

        if return_path:
            path_t = [0]
            path_p = [p]
            path_h = [h]

        if smoother:
            class_loss_vec = torch.zeros(cov.shape[0], device = h.device)
            num_evals_vec  = torch.zeros(cov.shape[0], device = h.device)
            assert class_criterion is not None

        assert len(times) + 1 == len(time_ptr)
        assert (len(times) == 0) or (times[-1] <= T)

        eval_times_total = torch.tensor([], dtype=torch.float64, device=h.device)
        eval_vals_total  = torch.tensor([], dtype=torch.float32, device=h.device)

        for i, obs_time in enumerate(times):
            ## Propagation of the ODE until next observation
            frac_of_delta_t = .001  # for numerical consistancy
            while current_time < (obs_time - frac_of_delta_tr*delta_t):
                 
                if self.solver == "dopri5":
                    h, p, current_time, eval_times, eval_ps = self.ode_step(h, p, obs_time-current_time, current_time)
                else:
                    h, p, current_time, eval_times, eval_ps = self.ode_step(h, p, delta_t, current_time)
                eval_times_total = torch.cat((eval_times_total, eval_times))
                eval_vals_total  = torch.cat((eval_vals_total, eval_ps))

                #Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_p.append(p)
                    path_h.append(h)

            ## Reached an observation
            start = time_ptr[i]
            end = time_ptr[i+1]

            X_obs = X[start:end]
            M_obs = M[start:end]
            i_obs = obs_idx[start:end]

            ## Using GRUObservationCell to update h. Also updating p and loss
            h, losses = self.gru_obs(h, p, X_obs, M_obs, i_obs)
           
            if smoother:
                class_loss_vec[i_obs] += class_criterion(
                    self.classification_model(h[i_obs]),labels[i_obs]).squeeze(1)
                num_evals_vec[i_obs] +=1
            prejump_loss = prejump_loss + losses.sum()
            p = self.p_model(h)

            postjump_loss += compute_KL_loss(p_obs=p[i_obs], X_obs=X_obs, 
                                             M_obs=M_obs, logvar=self.use_logvar)

            if return_path:
                path_t.append(obs_time)
                path_p.append(p)
                path_h.append(h)

        ## after every observation has been processed, propagating until T
        while current_time < T:
            timestep
            if self.solver == "dopri5":
                timestep = T-current_time
            else:
                timestep = delta_t
            h, p, current_time, eval_times, eval_ps = self.ode_step(h, p, 
                    timestep, current_time)
            eval_times_total = torch.cat((eval_times_total,eval_times))
            eval_vals_total  = torch.cat((eval_vals_total, eval_ps))
            
            #Storing the predictions
            if return_path:
                path_t.append(current_time)
                path_p.append(p)
                path_h.append(h)

        loss = prejump_loss + self.mixing * postjump_loss

        if smoother:
            class_loss_vec += class_criterion(self.classification_model(h),labels).squeeze(1)
            class_loss_vec /= num_evals_vec
        
        class_pred = self.classification_model(h)
       
        results = [h, loss, class_pred]
        if return_path:
            results.extend([np.array(path_t), torch.stack(path_p), torch.stack(path_h)])
            if smoother:
                results.extend([class_loss_vec])
            else:
                results.extend([eval_times_total, eval_vals_total])
        else:
            if smoother:
                results.extend([class_loss_vec])
            else:
                results.extend([prejump_loss])
        return results


def compute_KL_loss(p_obs: torch.Tensor, X_obs: torch.Tensor, 
                    M_obs: torch.Tensor, obs_noise_std: float=1e-2, 
                    logvar:bool=True) -> float:
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
    obs_noise_std = torch.tensor(obs_noise_std)
    mean, var = torch.chunk(p_obs, 2, dim=1)

    if logvar:
        std = torch.exp(0.5*var)
    else:
        ## making var non-negative and also non-zero (by adding a small value)
        std = torch.pow(torch.abs(var) + 1e-5, 0.5)

    def gaussian_KL(mu_1, mu_2, sigma_1, sigma_2):
        return (torch.log(sigma_2) - torch.log(sigma_1) 
        + (torch.pow(sigma_1,2)+torch.pow((mu_1 - mu_2),2)) / (2*sigma_2**2) - 0.5)

    return ((gaussian_KL(mu_1=mean, mu_2=X_obs, sigma_1=std, sigma_2=obs_noise_std)
            * M_obs)
            .sum())

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
        self.gru_c = _GRUODECell(input_size=2*input_size, hidden_size=hidden_size,
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
    pass