from .registry import model_entrypoints
from .registry import is_model

from .atmodel_head import *


def build_atmodel_head(config, *args, **kwargs):
    model_name = config['MODEL']['HEAD']
    if not is_model(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    body = model_entrypoints(model_name)(config, *args, **kwargs)
    return body