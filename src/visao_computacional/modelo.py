"""Definição do modelo (MobileNetV2 + transfer learning), transforms e Dataset.

Pré-processamento:
- treino: RandomResizedCrop(224) + flip + rotação + ColorJitter (data augmentation);
- validação/teste/inferência: Resize(256) + CenterCrop(224);
- normalização com média/desvio do ImageNet (exigido pelos pesos pré-treinados).
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms

from src.machine_learning.utils import RAIZ

CLASSES: list[str] = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
TAMANHO: int = 224
MEDIA_IMAGENET: list[float] = [0.485, 0.456, 0.406]
DESVIO_IMAGENET: list[float] = [0.229, 0.224, 0.225]
CAMINHO_MODELO: Path = RAIZ / "modelos" / "visao_mobilenetv2.pt"


def transform_treino() -> transforms.Compose:
    """Transforms com data augmentation (usado SOMENTE no treino)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(TAMANHO, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA_IMAGENET, DESVIO_IMAGENET),
    ])


def transform_avaliacao() -> transforms.Compose:
    """Transforms determinísticos para validação, teste e inferência."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(TAMANHO),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA_IMAGENET, DESVIO_IMAGENET),
    ])


class DatasetResiduos(Dataset):
    """Dataset PyTorch a partir de listas de caminhos (relativos à raiz) e classes."""

    def __init__(self, caminhos: list[str], classes: list[str], transform: transforms.Compose) -> None:
        self.caminhos = caminhos
        self.alvos = [CLASSES.index(c) for c in classes]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.caminhos)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(RAIZ / self.caminhos[idx]).convert("RGB")
        return self.transform(img), self.alvos[idx]


def criar_modelo(pretreinado: bool = True, n_classes: int = len(CLASSES)) -> nn.Module:
    """MobileNetV2 (ImageNet) com nova cabeça de classificação para 6 classes."""
    pesos = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretreinado else None
    modelo = models.mobilenet_v2(weights=pesos)
    modelo.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(modelo.last_channel, n_classes),
    )
    return modelo


def congelar_base(modelo: nn.Module) -> None:
    """Fase 1: congela todo o extrator de características (só a cabeça treina)."""
    for p in modelo.features.parameters():
        p.requires_grad = False


def descongelar_ultimas(modelo: nn.Module, n_blocos: int = 5) -> None:
    """Fase 2 (fine-tuning): libera os últimos `n_blocos` blocos de `features`."""
    for bloco in list(modelo.features.children())[-n_blocos:]:
        for p in bloco.parameters():
            p.requires_grad = True
