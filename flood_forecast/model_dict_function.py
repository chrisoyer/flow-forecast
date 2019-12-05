from flood_forecast.transformer_xl.multi_head_base import MultiAttnHeadSimple
from flood_forecast.transformer_xl.transformer_basic import SimpleTransformer
from flood_forecast.transformer_xl.transformer_xl import TransformerXL
from torch.optim import Adam, SGD
from torch.nn import MSELoss, SmoothL1Loss, PoissonNLLLoss
import torch

pytorch_model_dict = {"MultiAttnHeadSimple":MultiAttnHeadSimple, "SimpleTransformer":SimpleTransformer, 
"TransformerXL":TransformerXL
}

pytorch_criterion_dict = {"MSE": MSELoss(), "SmoothL1Loss":SmoothL1Loss(), "PoissonNLLLoss":PoissonNLLLoss()}

pytorch_opt_dict = {"Adam":Adam, "SGD":SGD}

scikit_dict = {}

def generate_square_subsequent_mask(sz):
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask