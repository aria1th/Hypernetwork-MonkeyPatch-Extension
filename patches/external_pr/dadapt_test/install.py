

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


def get_dadapt_adam():
    if install_or_import():
        try:
            from dadaptation.dadapt_adam import DAdaptAdam
            return DAdaptAdam
        except (ModuleNotFoundError, ImportError):
            print('Cannot use DAdaptAdam!')
            from torch.optim import AdamW
            return AdamW