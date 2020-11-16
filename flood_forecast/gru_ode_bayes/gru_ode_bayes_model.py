import numpy as np
import torch
import gru_ode_bayes_model
import gru_ode_bayes_data_utils as data_utils
from typing import Optional

class GRU_ODE_Bayes_Classifier(object):
    """
    """
    def __init__(self, hidden_size: int=50, p_hidden: int=25, 
             prep_hidden: int=10, use_logvar: bool=True, mixing: float=1e-4, 
             delta_t: float=0.1, lambda_: float=0, #Weighting between classification and MSE loss.
             classification_hidden: int=2, cov_hidden: int=50, 
             weight_decay: float=0.0001, dropout_rate:float=0.2, 
             lr: float=0.001, full_gru_ode=True, no_cov: bool=True, 
             impute:bool=False, verbose: int=0, #from 0 to 3 (highest)
             T_val: int=150, max_val_samples: int=3, solver:str="euler",
             device: Optional[str]=None, T:int=50
            ) -> None:
        """ The full GRU ODE Bayes model. Classifies time series (single series
        or sets of vectors)
        
        Usage:
        gru_ode_b = GRU_ODE_Bayes(**model_params)
        gru_ode_b.fit(**data)
        preds = gru_ode_b.predict(**new_data)

        Parameters:
        ____________
        ~Model Parameters~: Passed to create model
            hidden_size:   width of hidden layer in several locations
            p_hidden:      width of second layer down
            prep_hidden:   something to do with GRU obvservation cell
            use_logvar:
            mixing:        hyperp for prejump_loss and postjump_loss. 
                           loss = prejump_loss + mixing*postjump_loss
            delta_t:       step size for evalutating ODEs. smaller->more accurate
            T:             total time
            lambda_        
            classification_hidden
            cov_hidden
            dropout_rate
            full_gru_ode   use full gru_ode cells
            no_cov
            impute
            T_val
            max_val_samples
            solver         solver to use for ODEs, ['euler', 'midpoint', dorpi5']
        ~Driver Parameters~: used for running model
            lr: float, learning rate
            weight_decay: float, decay of lr over epochs
            device: str, cpu or CUDA
            verbose: int, 1, 2, 3
        
        """
        self.hidden_size = hidden_size
        self.p_hidden = p_hidden
        self.use_logvar = use_logvar
        self.mixing = mixing
        self.delta_t = delta_t
        self.lambda_ = lambda_
        self.classification_hidden = classification_hidden
        self.cov_hidden = cov_hidden
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.full_gru_ode = full_gru_ode
        self.no_cov = no_cov
        self.impute = impute
        self.verbose = verbose
        self.T_val = T_val
        self.max_val_samples = max_val_samples
        self.solver = solver
        self.device = device if device else torch.device(
                            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.delta_t = delta_t
        self.model_attributes = ["hidden_size", "p_hidden", "prep_hidden",
             "use_logvar", "mixing", "delta_t", "T", "lambda_",
             "classification_hidden", "cov_hidden", "dropout_rate", 
             "full_gru_ode", "no_cov", "impute", "T_val", 
             "max_val_samples" "solver"] # clump these to pass to NNFOwithBayesianJumps
        self.model = (NNFOwithBayesianJumps({attr: eval(f"self.{attr}") 
                                           for attr in self.model_attributes})
                                           .to(self.device))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                            weight_decay=self.weight_decay)


    def _load_batch(self, include_val:bool=False) -> None:
        """loads another batch of records into data_load attribute, for dumping
        into model"""
        self.data_load = {'times': self.raw_dload["times"], 
            'time_indices': self.raw_dload["time_indices"],
            'X': self.raw_dload["X"].to(self.device), 
            'M': self.raw_dload["M"].to(self.device),
            'obs_idx': self.raw_dload["obs_idx"], 
            'delta_t': self.raw_dload["delta_t"], 
            'T': self.raw_dload["T"], 'cov': self.raw_dload["cov"].to(self.device)}
        if include_val:
            self.data_load.update({'X_val': self.raw_dload["X_val"].to(device), 
                'M_val': self.raw_dload["M_val"].to(device), 
                'times_val': self.raw_dload["times_val"],
                'times_idx': self.raw_dload["index_val"]})
    
    def _extract_from_path(self, t_vec:np.array, p_vec:np.array, eval_times:np.array,
                           path_idx_eval:int) -> np.array:
        '''
        Arguments :
            t_vec :      vector of absolute times length [T]. Should be ordered.
            p_vec :      array of means and logvars of a trajectory at times 
                           t_vec. [T x batch_size x (2xfeatures)]
            eval_times: vector of absolute times at which we want to retrieve p_vec. [L]
            path_idx_eval : index of trajectory that we want to retrieve. Should be same length of eval_times. [L]
        Returns :
            Array of dimensions [L,(2xfeatures)] of means and logvar of the required eval times and trajectories
        '''
        def _map_to_closest(input, reference):
            output = np.zeros_like(input)
            for idx, element in enumerate(input):
                closest_idx=np.abs(reference-element).argmin()
                output[idx]=reference[closest_idx]
            return(output)
        #Remove the evaluation after the updates. Only takes the prediction before the Bayesian update. 
        t_vec, unique_index = np.unique(t_vec, return_index=True)
        p_vec = p_vec[unique_index,:,:]
        present_mask = np.isin(eval_times, t_vec)
        eval_times[~present_mask] = _map_to_closest(eval_times[~present_mask],t_vec)
        mapping = dict(zip(t_vec,np.arange(t_vec.shape[0])))
        time_idx = np.vectorize(mapping.get)(eval_times)

        return p_vec[time_idx, path_idx_eval, :]



    def fit(self, data_source: str=None, X: np.array=None, #X_val: Optional[np.array], 
            val_frac: Optional[float]=None,
             epoch_max:int=200) -> object:
        """fits model to data
        Arguments:

        Returns:
            self
        """
        self.data_source = data_source
        self.raw_dload = torch.utils.data.DataLoader(dataset=self.data_source,
            collate_fn=data_utils.custom_collate_fn, shuffle=True, 
            batch_size=100, num_workers=4)
        self.epoch_max = epoch_max
        for epoch in epoch_max:
            model.train()
            optimizer.zero_grad()
            for i, data in tqdm.tqdm(enumerate(self.raw_dload)):
                self._load_batch()
                self.hT, self.loss, _, _  = self.model(**self.data_load)
                loss.backward()
                if i%10==0:
                    optimizer.step()
                    optimizer.zero_grad()

            # validation
            with torch.no_grad():
                # todo: make into vectors so entire run can be inspected
                self.mse_val = self.loss_val = self.num_obs = 0
                model.eval()
                for i, b in enumerate(self.raw_dload):
                    self._load_batch(include_val=True)
                    # does this even get used??

                    (hT, loss, _, t_vec, p_vec, h_vec, eval_times, 
                    _) = model(**self.data_load, return_path=True)
                    #  Round floating points error in the time vector to match sig digs in delta_t
                    t_vec = (np.around(a=t_vec, decimals=str(delta_t)[::-1].find('.'))
                        .astype(np.float32))

                    p_val     = self._extract_from_path(t_vec, p_vec, times_val, times_idx)
                    m, v      = torch.chunk(p_val,2, dim=1)
                    last_loss = (data_utils.log_lik_gaussian(X_val, m, v)*M_val).sum()
                    mse_loss  = (torch.pow(X_val-m, 2) * M_val).sum()

                    self.loss_val += last_loss.cpu().numpy()
                    self.mse_val  += mse_loss.cpu().numpy()
                    self.num_obs  += M_val.sum().cpu().numpy()

                self.loss_val /= self.num_obs
                self.mse_val  /= self.num_obs
                print(f"Mean validation loss at epoch {epoch}: "
                      f"neg log lik={self.loss_val:.5f}," 
                      f"mse={self.mse_val:.5f}  (num_obs={self.num_obs})")
        return self
    
    
    def predict(self, X: np.array) -> np.array:
        """uses fitted model to predict
        Arugments:
        
        Returns:
            predictions: np.array
        """
        return predictions
    
    def save_model(self, file_path: str=None) -> None:
        """save model object to file_path. should be <filename>.pt"""
        torch.save(self.model.state_dict(), file_path)
        print(f"Saved model to '{file_path}'.")