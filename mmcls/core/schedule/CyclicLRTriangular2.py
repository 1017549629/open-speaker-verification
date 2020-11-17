from mmcv.runner.hooks.lr_updater import CyclicLrUpdaterHook
from math import cos, pi
from mmcv.utils import Registry

HOOKS = Registry('hook')


@HOOKS.register_module()
class CyclicTriLrUpdaterHook(CyclicLrUpdaterHook):
    def __init__(self,
                 by_epoch=False,
                 target_ratio=(10, 1e-4),
                 cyclic_times=1,
                 step_ratio_up=0.5,
                 method="tri2",
                 decay_ratio=0.5):
        super(CyclicTriLrUpdaterHook, self).__init__(by_epoch=by_epoch,
                                                     target_ratio=target_ratio,
                                                     cyclic_times=cyclic_times,
                                                     step_ratio_up=step_ratio_up)
        assert method in ["tri", "tri2", "cos"]
        self.method = method
        self.decay_ratio = decay_ratio

    def before_run(self, runner):
        super(CyclicLrUpdaterHook, self).before_run(runner)
        # initiate lr_phases
        # total lr_phases are separated as up and down
        max_iter_per_phase = runner.max_iters // self.cyclic_times
        iter_up_phase = int(self.step_ratio_up * max_iter_per_phase)
        # here remember target_ratio[1] -> target_ratio[0]
        self.lr_phases.append(
            [0, iter_up_phase, max_iter_per_phase, self.target_ratio[1], self.target_ratio[0]])
        self.lr_phases.append([
            iter_up_phase, max_iter_per_phase, max_iter_per_phase,
            self.target_ratio[0], self.target_ratio[1]
        ])

    def get_lr(self, runner, base_lr):
        curr_iter = runner.iter
        cycle_num = curr_iter // (runner.max_iters // self.cyclic_times)
        for (start_iter, end_iter, max_iter_per_phase, start_ratio,
             end_ratio) in self.lr_phases:
            curr_iter %= max_iter_per_phase
            if start_iter <= curr_iter < end_iter:
                progress = curr_iter - start_iter
                if self.method == "cos":
                    return annealing_cos(base_lr * start_ratio,
                                         base_lr * end_ratio,
                                         progress / (end_iter - start_iter))
                elif self.method == "tri":
                    return annealing_linear(base_lr * start_ratio,
                                            base_lr * end_ratio,
                                            progress / (end_iter - start_iter))
                elif self.method == "tri2":
                    decay = self.decay_ratio ** cycle_num if cycle_num != 0 else 1.0
                    return annealing_linear(base_lr * start_ratio * decay,
                                            base_lr * end_ratio * decay,
                                            progress / (end_iter - start_iter))


def annealing_cos(start, end, factor, weight=1):
    """Calculate annealing cos learning rate.
    Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
    percentage goes from 0.0 to 1.0.
    Args:
        start (float): The starting learning rate of the cosine annealing.
        end (float): The ending learing rate of the cosine annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
        weight (float, optional): The combination factor of `start` and `end`
            when calculating the actual starting learning rate. Default to 1.
    """
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out


def annealing_linear(start, end, factor):
    return start + (end-start) * factor


if __name__ == "__main__":
    hook = CyclicTriLrUpdaterHook(cyclic_times=3, target_ratio=(1, 1e-5), decay_ratio=0.7)
    class ABC:

        def __init__(self, max_iters):
            self.iter = 0
            self.max_iters = max_iters
            self.optimizer = {}

    runner = ABC(12000*3)
    hook.before_run(runner)
    runner.iter = 20000
    print(hook.get_lr(runner, 0.001))