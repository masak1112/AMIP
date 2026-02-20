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

def assemble_forcing(forcing, invariant):
    out = torch.cat((forcing, invariant), dim=-1) # b h w c
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

def disassemble_prognostic_forcing(x, nsurface=9, ndiagnostic=9, nlevels=26, nforcing=3, ninvariant=2):
    x = rearrange(
        x, "b c h w -> b h w c"
    )

    surface = x[..., : nsurface]
    diagnostic = x[..., nsurface : nsurface + ndiagnostic]
    forcing = x[..., nsurface + ndiagnostic : nsurface + ndiagnostic + nforcing]
    invariant = x[..., nsurface + ndiagnostic + nforcing : nsurface + ndiagnostic + nforcing + ninvariant]
    multilevel = x[..., nsurface + ndiagnostic + nforcing + ninvariant :]

    multilevel = rearrange(
        multilevel,
        "b h w (l c) -> b l h w c",
        l=nlevels,
    )

    return surface, diagnostic, forcing, invariant, multilevel

def disassemble_forcing(x, nforcing=3, ninvariant=2):
    x = rearrange(
        x, "b c h w -> b h w c"
    )

    forcing = x[..., : nforcing]
    invariant = x[..., nforcing : nforcing + ninvariant]

    return forcing, invariant

def fix_state_dict(state_dict, prefix="decoder."):
    return {
        k[len(prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }


    
    
