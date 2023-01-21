import torch
import ldm.models.diffusion.ddpm
from modules import shared


class Scheduler:
    """ Proportional Noise Step Scheduler"""
    def __init__(self, cycle_step=128, repeat=True):
        self.disabled = True
        self.cycle_step = int(cycle_step)
        self.repeat = repeat
        self.run_assertion()

    def __call__(self, value, step):
        if self.disabled:
            return value
        if self.repeat:
            step %= self.cycle_step
            return max(1, int(value * step / self.cycle_step))
        else:
            return value if step >= self.cycle_step else max(1, int(value * step / self.cycle_step))

    def run_assertion(self):
        assert type(self.cycle_step) is int
        assert type(self.repeat) is bool
        assert not self.repeat or self.cycle_step > 0

    def set(self, cycle_step=-1, repeat=-1, disabled=True):
        self.disabled = disabled
        if cycle_step >= 0:
            self.cycle_step = int(cycle_step)
        if repeat != -1:
            self.repeat = repeat
        self.run_assertion()


training_scheduler = Scheduler(cycle_step=-1, repeat=False)


def get_current(value, step=None):
    if step is None:
        if hasattr(shared, 'accessible_hypernetwork'):
            hypernetwork = shared.accessible_hypernetwork
        else:
            return value
        if hasattr(hypernetwork, 'step') and hypernetwork.training and hypernetwork.step is not None:
            return training_scheduler(value, hypernetwork.step)
        return value
    return max(1, training_scheduler(value, step))


def set_scheduler(cycle_step, repeat, enabled=False):
    global training_scheduler
    training_scheduler.set(cycle_step, repeat, not enabled)


def forward(self, x, c, *args, **kwargs):
    t = torch.randint(0, get_current(self.num_timesteps), (x.shape[0],), device=self.device).long()
    if self.model.conditioning_key is not None:
        assert c is not None
        if self.cond_stage_trainable:
            c = self.get_learned_conditioning(c)
        if self.shorten_cond_schedule:  # TODO: drop this option
            tc = self.cond_ids[t].to(self.device)
            c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
    return self.p_losses(x, c, t, *args, **kwargs)




ldm.models.diffusion.ddpm.LatentDiffusion.forward = forward
