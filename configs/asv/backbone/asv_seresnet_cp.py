# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SEResNet_SV',
        in_channels=1,
        stem_channels=32,
        base_channels=32,
        depth=34,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
	with_cp=True,
        norm_cfg=dict(type='BN', requires_grad=True, momentum=0.5)
    ),
    neck=dict(
        type='StatsPooling',
        emb_dim=256,
        in_plane=2816
    ),
    head=dict(
        type='AMSoftmaxClsHead',
        num_classes=5994,
        in_channels=256,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,5)
    )
)


if __name__ == "__main__":
    import torch
    from mmcls.models import build_backbone, build_neck, build_classifier
    with torch.no_grad():
        model = build_classifier(model)
        print(model)
        input = torch.rand(2, 200, 81)
        gt_label = torch.LongTensor([1, 2])
        print(model(input, gt_label=gt_label))
