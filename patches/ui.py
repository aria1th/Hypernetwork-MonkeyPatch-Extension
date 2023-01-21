import os

from modules import shared
from .hypernetwork import Hypernetwork, load_hypernetwork


def create_hypernetwork_load(name, enable_sizes, overwrite_old, layer_structure=None, activation_func=None, weight_init=None, add_layer_norm=False, use_dropout=False, dropout_structure=None, optional_info=None,
                        weight_init_seed=None, normal_std=0.01, skip_connection=False):
    # Remove illegal characters from name.
    name = "".join( x for x in name if (x.isalnum() or x in "._- "))
    assert name, "Name cannot be empty!"
    fn = os.path.join(shared.cmd_opts.hypernetwork_dir, f"{name}.pt")
    if not overwrite_old:
        assert not os.path.exists(fn), f"file {fn} already exists"

    if type(layer_structure) == str:
        layer_structure = [float(x.strip()) for x in layer_structure.split(",")]

    if dropout_structure and type(dropout_structure) == str:
        dropout_structure = [float(x.strip()) for x in dropout_structure.split(",")]
    normal_std = float(normal_std)
    assert normal_std > 0, "Normal Standard Deviation should be bigger than 0!"
    hypernet = Hypernetwork(
        name=name,
        enable_sizes=[int(x) for x in enable_sizes],
        layer_structure=layer_structure,
        activation_func=activation_func,
        weight_init=weight_init,
        add_layer_norm=add_layer_norm,
        use_dropout=use_dropout,
        dropout_structure=dropout_structure if use_dropout and dropout_structure else [0] * len(layer_structure),
        optional_info=optional_info,
        generation_seed=weight_init_seed if weight_init_seed != -1 else None,
        normal_std=normal_std,
        skip_connection=skip_connection
    )
    hypernet.save(fn)
    shared.reload_hypernetworks()
    hypernet = load_hypernetwork(name)
    assert hypernet is not None, f"Cannot load from {name}!"
    return hypernet


def create_hypernetwork(name, enable_sizes, overwrite_old, layer_structure=None, activation_func=None, weight_init=None, add_layer_norm=False, use_dropout=False, dropout_structure=None, optional_info=None,
                        weight_init_seed=None, normal_std=0.01, skip_connection=False):
    # Remove illegal characters from name.
    name = "".join( x for x in name if (x.isalnum() or x in "._- "))
    assert name, "Name cannot be empty!"
    fn = os.path.join(shared.cmd_opts.hypernetwork_dir, f"{name}.pt")
    if not overwrite_old:
        assert not os.path.exists(fn), f"file {fn} already exists"

    if type(layer_structure) == str:
        layer_structure = [float(x.strip()) for x in layer_structure.split(",")]

    if dropout_structure and type(dropout_structure) == str:
        dropout_structure = [float(x.strip()) for x in dropout_structure.split(",")]
    normal_std = float(normal_std)
    assert normal_std >= 0, "Normal Standard Deviation should be bigger than 0!"
    hypernet = Hypernetwork(
        name=name,
        enable_sizes=[int(x) for x in enable_sizes],
        layer_structure=layer_structure,
        activation_func=activation_func,
        weight_init=weight_init,
        add_layer_norm=add_layer_norm,
        use_dropout=use_dropout,
        dropout_structure=dropout_structure if use_dropout and dropout_structure else [0] * len(layer_structure),
        optional_info=optional_info,
        generation_seed=weight_init_seed if weight_init_seed != -1 else None,
        normal_std=normal_std,
        skip_connection=skip_connection
    )
    hypernet.save(fn)

    shared.reload_hypernetworks()

    return name, f"Created: {fn}", ""
