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
        multilevel, "b l h w c -> b h w (l c)"
    )
    if diagnostic is None:
        out = torch.cat((surface, multilevel), dim=-1) # b h w c
    else:
        out = torch.cat((surface, diagnostic, multilevel), dim=-1) # b h w c
    out = rearrange(
        out, "b h w c -> b c h w"
    )

    return out

def disassemble_input(x, nsurface=6, ndiagnostic=9, nlevels=26):
    x = rearrange(
        x, "b c h w -> b h w c"
    )

    surface = x[..., : nsurface]
    diagnostic = x[..., nsurface : nsurface + ndiagnostic]
    multilevel = x[..., nsurface + ndiagnostic :]

    multilevel = rearrange(
        multilevel,
        "b h w (l c) -> b l h w c",
        l=nlevels,
    )

    return surface, multilevel, diagnostic
    
    
