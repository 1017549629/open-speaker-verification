import os.path as osp
import torch

from mmcv.runner import Hook
from torch.utils.data import DataLoader


class ReduceLROnPlateauHook(Hook):
    """ReduceLROnPlateau hook.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by iters). Default: 10000.
        patience (int): bad trial limit, default 2.
        gamma (float): lr rescale factor, default 0.1.
    """

    def __init__(self, dataloader, interval=10000, patience=2, gamma=0.1, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got'
                            f' {type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.eval_kwargs = eval_kwargs
        self.best_acc = 0.0
        self.patience = patience
        self.gamma = gamma
        self.bad_trial = 0

    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return
        from mmcls.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        self.check_update(eval_res.get("top-1"), runner)
        print("Eval Metric: \n",
              "top-1 : {}, top-5 : {}, best top-1 : {}, bad trial number : {}".format(
                  eval_res.get("top-1"), eval_res.get("top-5"), self.best_acc, self.bad_trial))

        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

    def _set_lr(self, runner, gamma):
        if isinstance(runner.optimizer, dict):
            for k, optim in runner.optimizer.items():
                for param_group in optim.param_groups:
                    param_group['lr'] *= gamma
        else:
            for param_group in runner.optimizer.param_groups:
                param_group['lr'] *= gamma

    def check_update(self, top1, runner):
        if top1 > self.best_acc:
            # when better result is achieved, clean bad trial, update acc
            self.best_acc = top1
            self.bad_trial = 0
        else:
            self.bad_trial += 1

        if self.bad_trial > self.patience:
            print("bad trial number surpasses limit, decay lr")
            self._set_lr(runner, self.gamma)
            self.bad_trial = 0


class DistReduceLROnPlateauHook(ReduceLROnPlateauHook):
    """Distributed reduce lr on plateau hook.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by iters). Default: 10000.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self,
                 dataloader,
                 interval=10000,
                 patience=2,
                 gamma=0.1,
                 gpu_collect=False,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got '
                            f'{type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.eval_kwargs = eval_kwargs
        self.best_acc = 0.0
        self.patience = patience
        self.gamma = gamma
        self.bad_trial = 0

    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return
        from mmcls.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.DistReduceLR_eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)
