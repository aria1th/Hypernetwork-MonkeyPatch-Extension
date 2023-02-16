

def install_or_import() -> bool:
    try:
        import pip
        try:
            import dadaptation
        except (ModuleNotFoundError, ImportError):
            print("Trying to install dadaptation...")
            pip.main(['install', 'dadaptation'])
            return True
    except (ModuleNotFoundError, ImportError):
        print("Cannot found pip!")
        return False
    return True


def get_dadapt_adam(optimizer_name=None):
    if install_or_import():
        if optimizer_name is None or optimizer_name in ['DAdaptAdamW', 'AdamW', 'DAdaptAdam', 'Adam']:  # Adam-dadapt implementation
            try:
                from dadaptation.dadapt_adam import DAdaptAdam
                return DAdaptAdam
            except (ModuleNotFoundError, ImportError):
                print('Cannot use DAdaptAdam!')
        elif optimizer_name == 'DAdaptSGD':
            try:
                from dadaptation.dadapt_sgd import DAdaptSGD
                return DAdaptSGD
            except (ModuleNotFoundError, ImportError):
                print('Cannot use DAdaptSGD!')
        elif optimizer_name == 'DAdaptAdagrad':
            try:
                from dadaptation.dadapt_adagrad import DAdaptAdaGrad
                return DAdaptAdaGrad
            except (ModuleNotFoundError, ImportError):
                print('Cannot use DAdaptAdaGrad!')
    from torch.optim import AdamW
    return AdamW
