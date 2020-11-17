from mmcv.utils import build_from_cfg


def get_lr_hook(lr_config, HOOKS):
    if isinstance(lr_config, dict):
        assert 'policy' in lr_config
        policy_type = lr_config.pop('policy')
        # If the type of policy is all in lower case, e.g., 'cyclic',
        # then its first letter will be capitalized, e.g., to be 'Cyclic'.
        # This is for the convenient usage of Lr updater.
        # Since this is not applicable for `
        # CosineAnnealingLrUpdater`,
        # the string will not be changed if it contains capital letters.
        if policy_type == policy_type.lower():
            policy_type = policy_type.title()
        hook_type = policy_type + 'LrUpdaterHook'
        lr_config['type'] = hook_type
        hook = build_from_cfg(lr_config, HOOKS)
    else:
        hook = lr_config
    return hook