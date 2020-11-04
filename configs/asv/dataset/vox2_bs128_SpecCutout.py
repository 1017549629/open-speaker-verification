# dataset settings
dataset_type = 'SpeakerDataset'

train_pipeline = [
    # dict(type='ToTensor', keys=['gt_label']),
    dict(type='ToTensor', keys=['img']),
    dict(type='SpecCutout', keys=['img'], f_mask=8, t_mask=10),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    # dict(type='ToTensor', keys=['gt_label']),
    dict(type='ToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type, data_prefix='/data1/mayufeng/data_vox/train',
        pipeline=train_pipeline, map="/data1/mayufeng/data_vox/train/uid2classes.json"),
    val=dict(
        type=dataset_type, data_prefix='/data1/mayufeng/data_vox/val',
        pipeline=test_pipeline, map="/data1/mayufeng/data_vox/train/uid2classes.json"),
    test=dict(
        type=dataset_type, data_prefix='/data1/mayufeng/data_vox/val',
        pipeline=test_pipeline, map="/data1/mayufeng/data_vox/train/uid2classes.json"))


if __name__ == "__main__":
    from mmcls.datasets import build_dataset
    dataset = build_dataset(data["train"])
    dataset.test_mode = True
    print(dataset[1000])