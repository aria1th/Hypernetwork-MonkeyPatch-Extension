
from modules.shared import cmd_opts, opts
import modules.shared


def reload_hypernetworks():
    from .hypernetwork import list_hypernetworks, load_hypernetwork
    modules.shared.hypernetworks = list_hypernetworks(cmd_opts.hypernetwork_dir)
    load_hypernetwork(opts.sd_hypernetwork)


try:
    modules.shared.reload_hypernetworks = reload_hypernetworks
except:
    pass
