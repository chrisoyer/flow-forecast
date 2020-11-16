from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

def custom_collate_fn(batch):
    """rearranges data for main model's ingestion
    batch source should have a column named <Time>
    Other columns should either start with Value or Mask, for data and masking data, respectively. 
    """
    idx2batch = pd.Series(np.arange(len(batch)), index = [b["idx"] for b in batch])

    def  _parser(batch):
        pat_idx   = [b["idx"] for b in batch]
        df = (pd.concat([b["path"] for b in batch], axis=0)
                 .sort_values(by=["Time"]))
        df_cov = torch.Tensor([b["cov"] for b in batch])
        labels = torch.tensor([b["y"] for b in batch])
        batch_ids = idx2batch[df.index].to_numpy()

    ## calculating number of events at every time
    times, counts = np.unique(df.Time.values, return_counts=True)
    time_indices = np.concatenate([[0], np.cumsum(counts)])

    ## tensors for the data in the batch
    value_cols = [c.startswith("Value") for c in df.columns]
    mask_cols  = [c.startswith("Mask") for c in df.columns]

    if batch[0]['val_samples'] is not None:
        df_after = (pd.concat(b["val_samples"] for b in batch)
                   .sort_values(by=["ID", "Time"]))
        value_cols_val = [c.startswith("Value") for c in df_after.columns]
        mask_cols_val  = [c.startswith("Mask") for c in df_after.columns]
        X_val = torch.tensor(df_after.iloc[:,value_cols_val].to_numpy())
        M_val = torch.tensor(df_after.iloc[:,mask_cols_val].to_numpy())
        times_val = df_after["Time"].to_numpy()
        index_val = idx2batch[df_after["ID"]].to_numpy()

        # Last observation before the T_val cut_off. THIS IS LIKELY TO GIVE ERRORS IF THE NUMBER OF VALIDATION SAMPLES IS HIGHER THAN 2. CHECK THIS.
        if batch[0]["store_last"]:
            df_last = df[~df.index.duplicated(keep="last")].copy()
            index_last = idx2batch[df_last.index].to_numpy()
            perm_last = sort_array_on_other(index_val, index_last)
            tens_last = torch.tensor(df_last.iloc[:,value_cols].values[perm_last,:])
            index_last = index_last[perm_last]
        else:
            index_last = tens_last = 0
    else:
        X_val = M_val = times_val = index_val = tens_last = index_last = None

    # Assemble data into dictionary
    return {
        "pat_idx": pat_idx,
        "times": times,
        "time_indices": time_indices,
        "X": df.iloc[:, value_cols].to_numpy(),
        "M": df.iloc[:, mask_cols].to_numpy(),
        "obs_idx": torch.tensor(batch_ids),
        "y": labels,
        "cov": df_cov,
        "X_val": X_val,
        "M_val": M_val,
        "times_val": times_val,
        "index_val": index_val,
        "X_last": tens_last,
        "obs_idx_last": index,
        }

def log_lik_gaussian(x, mu, logvar):
    return np.log(np.sqrt(2*np.pi)) + (logvar/2) + ((x-mu).pow(2)/(2*logvar.exp()))

def nan_to_mask():
    """creates mask from dataframe with NANs
    TODO: everything
    """
    pass

class ODE_Dataset(Dataset):
    """
    Dataset class for ODE type of data. With 2 values.
    Can be fed with either a csv file containg the dataframe or directly with a panda dataframe.
    One can further provide samples idx that will be used (for training / validation split purposes.)
    """
    def __init__(self, data_df: pd.DataFrame=None,
                cov_df: pd.DataFrame=None,
                label_df: pd.DataFrame=None, 
                t_mult: float=1.0, idx=None, jitter_time: float=0, 
                validation: bool=False, val_options: dict=None) ->None:
        """
        Formatting: All dfs should have an <ID> column and the label df should
        have a <label> column
        Args:
            panda_df        alternatively use pandas df instead of CSV file
            cov_df          covariate data in df form
            label_file      label for individuals
            t_mult          multiplier for time values (1.0 default)
            jitter_time     jitter size (0 means no jitter), to add randomly to Time.
                            Jitter is added before multiplying with t_mult
            validation      True if this dataset is for validation purposes
            val_options     validation dataset options.
                                    T_val : Time after which observations are considered as test samples
                                    max_val_samples : maximum number of test observations per trajectory.
        """
        self.validation = validation

        self.data_df = data_df
        assert self.data_df.columns[0]=="ID"

        self.cov_df = cov_df
        if self.cov_df is None:
            self.cov_df = (pd.DataFrame({"ID":self.data_df["ID"].unique()}
                           .assign(Cov=0)))
        assert self.cov_df.columns[0]=="ID"
        
        self.label_df = label_df
        if self.label_df is None:
            self.label_df = (pd.DataFrame({"ID": self.data_df['ID'].unique()})
                             .assign(label=0))
        assert self.label_df.columns[0]=="ID"
        assert self.label_df.columns[1]=="label"
        
        #If validation : consider only the data with a least one observation before T_val and one observation after:
        self.store_last = False
        if self.validation:
            df_beforeIdx = self.data_df.loc[self.data_df["Time"]<=val_options["T_val"],"ID"].unique()
            if val_options.get("T_val_from"): #Validation samples only after some time.
                df_afterIdx = self.data_df.loc[self.data_df["Time"]>=val_options["T_val_from"],"ID"].unique()
                self.store_last = True #Dataset get will return a flag for the collate to compute the last sample before T_val
            else:
                df_afterIdx = self.data_df.loc[self.data_df["Time"]>val_options["T_val"],"ID"].unique()
            
            valid_idx = np.intersect1d(df_beforeIdx, df_afterIdx)
            self.data_df = self.data_df.query('ID==@valid_idx')
            self.label_df = self.label_df.query('ID==@valid_idx')
            self.cov_df = self.cov_df.query('ID==@valid_idx')

        if idx is not None:  ## ???
            self.data_df = self.data_df.query('ID==@idx')
            map_dict= dict(zip(self.data_df["ID"].unique(),np.arange(self.data_df["ID"].nunique())))
            self.data_df["ID"] = self.data_df["ID"].map(map_dict) # Reset the ID index.

            self.cov_df = self.cov_df.query('ID==@idx')
            self.cov_df["ID"] = self.cov_df["ID"].map(map_dict) # Reset the ID index.

            self.label_df = self.label_df.query('ID==@idx')
            self.label_df["ID"] = self.label_df["ID"].map(map_dict) # Reset the ID index.


        assert self.cov_df.shape[0]==self.data_df["ID"].nunique()

        self.variable_num = sum([c.startswith("Value") for c in self.data_df.columns]) #number of variables in the dataset
        self.cov_dim = self.cov_df.shape[1]-1

        self.cov_df = self.cov_df.astype(np.float32).set_index("ID")
        self.label_df.set_index("ID",inplace=True)

        self.data_df['Time'] = self.data_df.Time * t_mult

        @staticmethod
        def _add_jitter(df:pd.DataFrame=None, jitter_time: float=1e-3):
            """Modifies Double OU dataset, so that observations with both dimensions
            are split. One is randomly shifted earlier by amount 'jitter_time'.
            ## TODO: fix so 
            """
            both = (df["Mask_1"] == 1.0) & (df["Mask_2"] == 1.0)
            df_single = df[both == False]
            df_both   = df[both]
            df_both1  = df_both.copy()
            df_both2  = df_both.copy()

            df_both1["Mask_2"] = 0.0
            df_both2["Mask_1"] = 0.0
            jitter = np.random.randint(2, size=df_both1.shape[0])
            df_both1["Time"] -= jitter_time * jitter
            df_both2["Time"] -= jitter_time * (1 - jitter)

            df_jit = pd.concat([df_single, df_both1, df_both2])
            ## make sure time is not negative:
            df_jit.Time.clip_lower(0.0, inplace=True)

            val_cols = [c for c in df.columns if 'Val' in c]
            mask_cols = [c for c in df.columns if 'Mas' in c]
            df[val_cols] = df[val_cols].astype(np.float32)
            df[mask_cols] = df[mask_cols].astype(np.float32)
            return df_jit


        if jitter_time != 0:
            self.data_df = add_jitter(self.data_df, jitter_time=jitter_time)

            
        else:
            self.data_df = self.data_df.astype(np.float32)

        if self.validation:
            assert val_options is not None, "Validation set options should be fed"
            self.df_before = self.data_df.query(f'Time<=@{self.val_options["T_val"]}').copy().sort_values('Time')
            self.df_after  = self.data_df.query(f'Time>@{self.val_options["T_val"]}').copy().sort_values('Time')
            if val_options.get("T_closest") is not None:
                df_after = (df_after.assign(Time_from_target=lambda x: (x.Time - val_options["T_closest"]).abs())
                            .sort_values(by=["Time_from_target","Value_0"], ascending=True)
                            .drop_duplicates(subset=["ID"], keep="first",)
                            .drop(columns = ["Time_from_target"]))
            else:
                self.df_after = self.df_after.groupby("ID").head(val_options["max_val_samples"])

            self.data_df = self.df_before #We remove observations after T_val
            self.df_after = (self.df_after
                            .assign(ID=lambda x: x.ID.astype(np.int))
                            .sort_values("Time"))
        else:
            self.df_after = None


        self.length = self.data_df["ID"].nunique()
        self.data_df['ID'] = self.data_df.ID.astype(np.int)
        self.data_df = self.data_df.set_index("ID").sort_values("Time")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        subset = self.data_df.loc[idx]
        if len(subset.shape)==1: #Don't ask me anything about this.
            subset = self.data_df.loc[[idx]]

        covs = self.cov_df.loc[idx].values
        tag  = self.label_df.loc[idx].astype(np.float32).values
        if self.validation :
            val_samples = self.df_after.loc[self.df_after["ID"]==idx]
        else:
            val_samples = None
        ## returning also idx to allow empty samples
        return {"idx":idx, "y": tag, "path": subset, "cov": covs, 
                "val_samples":val_samples, "store_last":self.store_last}
