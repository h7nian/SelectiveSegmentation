"""Model wrappers exposing one interface for every benchmark condition.

Both wrappers take images as float tensors in [0, 1] (normalization happens
inside the wrapper, since CLIPSeg and DeepLabV3 use different statistics)
and produce per-class probability maps aligned with the dataset's classes,
so a single training and evaluation loop serves every condition.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)
from transformers import (
    CLIPSegForImageSegmentation,
    CLIPSegProcessor,
    SegformerConfig,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

from selectseg.data import IGNORE_INDEX

CLIPSEG_CHECKPOINT = "CIDAS/clipseg-rd64-refined"
CLIPSEG_REVISION = "999e0328d9e10b484360c477313983f9afdd7050"
DEEPLABV3_WEIGHTS = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
SEGFORMER_CHECKPOINT = "nvidia/segformer-b2-finetuned-ade-512-512"
SEGFORMER_REVISION = "de01bae28967510f9ddd496c60a969357195400c"
CONDITION_NAMES = {
    "clipseg": ("clipseg-general", "clipseg-target"),
    "deeplabv3": ("deeplabv3-external", "deeplabv3-target"),
    "segformer": ("segformer-external", "segformer-target"),
}


class SegmentationModel(nn.Module):
    """Shared interface: dataset-space probabilities and a training loss.

    Subclasses set :attr:`image_size` (the square input resolution) and the
    ``mean``/``std`` normalization buffers.
    """

    image_size: int

    def predict_probs(self, images):
        """Return per-class probabilities of shape (B, num_classes, H, W)."""
        raise NotImplementedError

    def compute_loss(self, images, masks, prompt_indices):
        """Return the training loss for a batch of (S, S) class-index masks."""
        raise NotImplementedError

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _normalize(self, images):
        return (images - self.mean) / self.std


class CLIPSegModel(SegmentationModel):
    """CLIPSeg with a fixed per-dataset prompt list.

    Each prompt yields an independent sigmoid map. A foreground class takes
    the maximum over its prompts' maps, and background is one minus the
    strongest foreground, so a pixel is foreground only where some prompt
    scores above 0.5. Fine-tuning trains the lightweight decoder and keeps
    the CLIP encoders frozen, as in the CLIPSeg paper.
    """

    image_size = 352

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            CLIPSEG_CHECKPOINT,
            revision=CLIPSEG_REVISION,
            local_files_only=True,
        )
        self.model.clip.requires_grad_(False)
        processor = CLIPSegProcessor.from_pretrained(
            CLIPSEG_CHECKPOINT,
            revision=CLIPSEG_REVISION,
            local_files_only=True,
        )
        tokens = processor.tokenizer(
            list(spec.prompts), padding=True, return_tensors="pt"
        )
        image_processor = processor.image_processor
        self.register_buffer("prompt_ids", tokens.input_ids, persistent=False)
        self.register_buffer(
            "prompt_attention", tokens.attention_mask, persistent=False
        )
        self.register_buffer(
            "prompt_classes", torch.tensor(spec.prompt_classes), persistent=False
        )
        self.register_buffer(
            "mean", torch.tensor(image_processor.image_mean).view(3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std", torch.tensor(image_processor.image_std).view(3, 1, 1),
            persistent=False,
        )

    def _logits(self, pixel_values, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        logits = outputs.logits
        # (N, H, W) regardless of the batch-of-one squeeze in transformers.
        return logits.reshape(-1, *logits.shape[-2:])

    def compute_loss(self, images, masks, prompt_indices):
        logits = self._logits(
            self._normalize(images),
            self.prompt_ids[prompt_indices],
            self.prompt_attention[prompt_indices],
        )
        prompt_class = self.prompt_classes[prompt_indices].view(-1, 1, 1)
        targets = (masks == prompt_class).float()
        valid = (masks != IGNORE_INDEX).float()
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (loss * valid).sum() / valid.sum().clamp(min=1)

    def predict_probs(self, images):
        batch_size = images.shape[0]
        num_prompts = self.prompt_ids.shape[0]
        # Pair every image with every prompt: images repeat per prompt, and
        # the prompt table tiles across images.
        pixel_values = self._normalize(images).repeat_interleave(num_prompts, dim=0)
        input_ids = self.prompt_ids.repeat(batch_size, 1)
        attention_mask = self.prompt_attention.repeat(batch_size, 1)
        logits = self._logits(pixel_values, input_ids, attention_mask)
        prompt_probs = logits.sigmoid().view(
            batch_size, num_prompts, *images.shape[-2:]
        )
        foreground = torch.stack(
            [
                prompt_probs[:, [i for i, c in enumerate(self.spec.prompt_classes) if c == cls]].amax(dim=1)
                for cls in range(1, self.spec.num_classes)
            ],
            dim=1,
        )
        background = 1 - foreground.amax(dim=1, keepdim=True)
        return torch.cat([background, foreground], dim=1)

    def configure_optimizer(self, lr, weight_decay):
        return torch.optim.AdamW(
            self.trainable_parameters(), lr=lr, weight_decay=weight_decay
        )


class DeepLabV3Model(SegmentationModel):
    """DeepLabV3-ResNet50, either the COCO checkpoint or fine-tuned.

    When the dataset's classes differ from the checkpoint's (Pet), the
    external condition maps checkpoint classes onto dataset classes through
    the prompt table (pet = cat + dog), while fine-tuning replaces the
    classification heads instead.
    """

    image_size = 512

    def __init__(self, spec, finetuned, init_weights=True):
        super().__init__()
        self.spec = spec
        weights = DEEPLABV3_WEIGHTS if init_weights else None
        # weights_backbone=None stops torchvision from downloading ImageNet
        # backbone weights when weights is None; the fine-tuned checkpoint
        # loaded on top supplies every parameter anyway.
        self.model = deeplabv3_resnet50(
            weights=weights, weights_backbone=None, aux_loss=True
        )
        categories = DEEPLABV3_WEIGHTS.meta["categories"]
        matches_checkpoint = spec.num_classes == len(categories)
        if finetuned and not matches_checkpoint:
            self.model.classifier[4] = nn.Conv2d(256, spec.num_classes, 1)
            self.model.aux_classifier[4] = nn.Conv2d(256, spec.num_classes, 1)
        if finetuned or matches_checkpoint:
            self.class_map = None
        else:
            # Zero-shot transfer: dataset class c = sum of the checkpoint
            # classes its prompts name (checkpoint names have no spaces).
            self.class_map = [
                [
                    categories.index(prompt.replace(" ", ""))
                    for prompt, cls in zip(spec.prompts, spec.prompt_classes)
                    if cls == c
                ]
                for c in range(1, spec.num_classes)
            ]
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1),
            persistent=False,
        )

    def compute_loss(self, images, masks, prompt_indices):
        del prompt_indices  # text prompts only apply to CLIPSeg
        outputs = self.model(self._normalize(images))
        loss = F.cross_entropy(outputs["out"], masks, ignore_index=IGNORE_INDEX)
        aux = F.cross_entropy(outputs["aux"], masks, ignore_index=IGNORE_INDEX)
        return loss + 0.4 * aux

    def predict_probs(self, images):
        logits = self.model(self._normalize(images))["out"]
        probs = logits.softmax(dim=1)
        if self.class_map is None:
            return probs
        foreground = torch.stack(
            [probs[:, indices].sum(dim=1) for indices in self.class_map], dim=1
        )
        background = 1 - foreground.sum(dim=1, keepdim=True)
        return torch.cat([background.clamp(min=0), foreground], dim=1)

    def configure_optimizer(self, lr, weight_decay):
        return torch.optim.SGD(
            self.trainable_parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )


class SegFormerModel(SegmentationModel):
    """SegFormer-B2 target model with an ImageNet/ADE-pretrained backbone."""

    image_size = 512

    def __init__(self, spec, finetuned, init_weights=True):
        super().__init__()
        if not finetuned:
            raise ValueError("SegFormer is supported only as a target-adapted model")
        id2label = {index: name for index, name in enumerate(spec.class_names)}
        label2id = {name: index for index, name in enumerate(spec.class_names)}
        if init_weights:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                SEGFORMER_CHECKPOINT,
                revision=SEGFORMER_REVISION,
                num_labels=spec.num_classes,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
                local_files_only=True,
            )
        else:
            config = SegformerConfig.from_pretrained(
                SEGFORMER_CHECKPOINT,
                revision=SEGFORMER_REVISION,
                local_files_only=True,
            )
            config.num_labels = spec.num_classes
            config.id2label = id2label
            config.label2id = label2id
            self.model = SegformerForSemanticSegmentation(config)
        processor = SegformerImageProcessor.from_pretrained(
            SEGFORMER_CHECKPOINT,
            revision=SEGFORMER_REVISION,
            local_files_only=True,
        )
        self.register_buffer(
            "mean", torch.tensor(processor.image_mean).view(3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(processor.image_std).view(3, 1, 1), persistent=False
        )

    def _upsampled_logits(self, images):
        logits = self.model(pixel_values=self._normalize(images)).logits
        return F.interpolate(
            logits, size=images.shape[-2:], mode="bilinear", align_corners=False
        )

    def compute_loss(self, images, masks, prompt_indices):
        del prompt_indices
        return F.cross_entropy(
            self._upsampled_logits(images), masks, ignore_index=IGNORE_INDEX
        )

    def predict_probs(self, images):
        return self._upsampled_logits(images).softmax(dim=1)

    def configure_optimizer(self, lr, weight_decay):
        return torch.optim.AdamW(
            self.trainable_parameters(), lr=lr, weight_decay=weight_decay
        )


def build_model(name, spec, finetuned, init_weights=True):
    """Build a supported segmentation-model wrapper.

    ``finetuned`` selects the target-adapted architecture (e.g. a
    dataset-sized DeepLabV3 head). ``init_weights=False`` skips downloading
    the external DeepLabV3 checkpoint when a fine-tuned checkpoint will be
    loaded on top; CLIPSeg always starts from its pretrained checkpoint.
    """
    if name == "clipseg":
        return CLIPSegModel(spec)
    if name == "deeplabv3":
        return DeepLabV3Model(spec, finetuned=finetuned, init_weights=init_weights)
    if name == "segformer":
        return SegFormerModel(spec, finetuned=finetuned, init_weights=init_weights)
    raise ValueError(f"unknown model: {name}")
