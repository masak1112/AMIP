import yaml
import argparse
from einops import rearrange
import torch 

def get_yaml(path):
    with open(path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def save_yaml(config, path):
    with open(path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def assemble_input(surface, multilevel, diagnostic=None):
    multilevel = rearrange(
        multilevel, "b c l h w -> b (c l) h w"
    )
    if diagnostic is None:
        out = torch.cat((surface, multilevel), dim=1) # b c h w
    else:
        out = torch.cat((surface, diagnostic, multilevel), dim=1) # b c h w

    return out

def assemble_forcing(forcing, invariant):
    out = torch.cat((forcing, invariant), dim=1) # b c h w

    return out

def disassemble_input(x, nsurface=6, ndiagnostic=15, nlevels=26):
    # x in b c h w
    surface = x[:, : nsurface]
    diagnostic = x[:, nsurface : nsurface + ndiagnostic]
    multilevel = x[:, nsurface + ndiagnostic :]

    multilevel = rearrange(
        multilevel,
        "b (c l) h w -> b c l h w",
        l=nlevels,
    )

    return surface, multilevel, diagnostic

def disassemble_forcing(x, nforcing=3, ninvariant=2):
    # x in b c h w

    forcing = x[:, : nforcing]
    invariant = x[:, nforcing : nforcing + ninvariant]

    return forcing, invariant

def fix_state_dict(state_dict, prefix="decoder."):
    return {
        k[len(prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }


    
    
