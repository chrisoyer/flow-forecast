import torch
import gru_ode_bayes_model
from typing import Optional

class gru_ode_bayes(object):
    """
    """
    def __init__(self, "hidden_size": int=50, "p_hidden": int=25, 
             "prep_hidden": int=10, "use_logvar": bool=True, "mixing": float=1e-4, 
             "delta_t": float=0.1, "T"=200, "lambda_": float=0, #Weighting between classification and MSE loss.
             "classification_hidden": int=2, "cov_hidden": int=50, 
             "weight_decay": float=0.0001, "dropout_rate":float=0.2, 
             "lr": float=0.001, "full_gru_ode"=True, "no_cov": bool=True, 
             "impute":bool=False, "verbose" int=0, #from 0 to 3 (highest)
             "T_val": int=150, "max_val_samples": int=3, "solver":str="euler",
             "device":str="cpu", "delta_t":float=.05, "T":int=50
            ) -> None:
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
        self.device = device
        self.delta_t = delta_t
        self.model_attributes = ["hidden_size", "p_hidden", "prep_hidden",
             "use_logvar", "mixing", "delta_t", "T", "lambda_",
             "classification_hidden", "cov_hidden", 
             "dropout_rate", "lr", "full_gru_ode", "no_cov", "impute",
             "verbose", "T_val", "max_val_samples" "solver",
             "delta_t", "T"]
        self.model = NNFOwithBayesianJumps({attr: eval("self.{attr}") 
                                           for attr in self.model_attributes})
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                            weight_decay=self.weight_decay)


    def _load_batch(self, data: dict=self.raw_dload, include_val:bool=False) -> None:
        """loads another batch of records into data_load attribute, for dumping
        into model"""
        self.data_load = {'times': data["times"], 'time_indices': data["time_indices"],
            'X': data["X"].to(self.device), 'M': data["M"].to(self.device),
            'obs_idx': data["obs_idx"], 'delta_t'=data["delta_t"], 
            'T'=data["T"], 'cov'=data["cov"].to(self.device)}
        if include_val:
            self.data_load.update({'X_val': data["X_val"].to(device), 
                'M_val': data["M_val"].to(device), 'times_val': data["times_val"],
                'times_idx': data["index_val"]}

    def fit(self, X: np.array=None, y: np.array=None, X_val: Optional[np.array], 
        y_val: Optional[np.array]=None, val_frac: Optional[float],
         epoch_max:int=200) -> gru_ode_bayes:
        """fits model to data"""
        self.X = X
        self.raw_dload = torch.utils.data.DataLoader(dataset=X,
            collate_fn=data_utils.custom_collate_fn, shuffle=True, 
            batch_size=100, num_workers=4)
        self.y = y

        self.epoch_max = epoch_max
        for epoch in epoch_max:
            model.train()
            optimizer.zero_grad()
            for i, data in tqdm.tqdm(enumerate(self.X)):
                self._load_batch()
                self.hT, self.loss, _, _  = self.model(**self.data_load)

                loss.backward()
                if i%10==0:
                    optimizer.step()
                    optimizer.zero_grad()

            with torch.no_grad():
                mse_val  = 0
                loss_val = 0
                num_obs  = 0
                model.eval()
                for i, b in enumerate(dl_val):
                    self._load_batch(include_val=True)
                    # does this even get used??
                    y = self.raw_dload["y"]

                    hT, loss, _, t_vec, p_vec, h_vec, eval_times,_ = model(times, time_ptr, X, M, obs_idx, delta_t=delta_t, T=T, cov=cov, return_path=True)
                    t_vec = np.around(t_vec,str(delta_t)[::-1].find('.')).astype(np.float32) #Round floating points error in the time vector.

                    p_val     = data_utils.extract_from_path(t_vec,p_vec,times_val,times_idx)
                    m, v      = torch.chunk(p_val,2,dim=1)
                    last_loss = (data_utils.log_lik_gaussian(X_val,m,v)*M_val).sum()
                    mse_loss  = (torch.pow(X_val - m, 2) * M_val).sum()

                    loss_val += last_loss.cpu().numpy()
                    mse_val  += mse_loss.cpu().numpy()
                    num_obs  += M_val.sum().cpu().numpy()

                loss_val /= num_obs
                mse_val  /= num_obs
                print(f"Mean validation loss at epoch {epoch}: nll={loss_val:.5f}, mse={mse_val:.5f}  (num_obs={num_obs})")
                
        return self
    
    
    predict(x: np.array) -> np.arry:
        """uses fitted model to predict"""
        return predictions
