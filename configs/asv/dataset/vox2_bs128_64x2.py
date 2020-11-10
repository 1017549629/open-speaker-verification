# dataset settings
train_dataset_type = 'PKSpeakerDataset'
val_dataset_type = 'SpeakerDataset'

train_pipeline = [
    # dict(type='ToTensor', keys=['gt_label']),
    dict(type='ToTensor', keys=['img']),
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
        type=train_dataset_type, data_prefix='/data1/mayufeng/data_vox/train', P=64, K=2,
        pipeline=train_pipeline, map="/data1/mayufeng/data_vox/train/uid2classes.json"),
    val=dict(
        type=val_dataset_type, data_prefix='/data1/mayufeng/data_vox/val',
        pipeline=test_pipeline, map="/data1/mayufeng/data_vox/train/uid2classes.json"),
    test=dict(
        type=val_dataset_type, data_prefix='/data1/mayufeng/data_vox/val',
        pipeline=test_pipeline, map="/data1/mayufeng/data_vox/train/uid2classes.json"))


if __name__ == "__main__":
    from mmcls.datasets import build_dataset
    dataset = build_dataset(data["train"])
    dataset.test_mode = True
    print(hasattr(dataset, "P"))
    print(dataset[1])