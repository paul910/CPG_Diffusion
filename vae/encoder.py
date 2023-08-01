from abc import ABC, abstractmethod, ABCMeta
from typing import Tuple, Dict, List

import torch

from vae.helper_types import Graph, Loss
from vae.helper_types import LatentRepresentation, GraphLatentRepresentation, NodesLatentRepresentation


class Encoder(torch.nn.Module, ABC):
    @abstractmethod
    def encode(self, graph: Graph) -> LatentRepresentation:
        pass

    @abstractmethod
    def loss(self, graph: Graph) -> Tuple[LatentRepresentation, Dict[str, Loss]]:
        pass

    @abstractmethod
    def loss_keys(self) -> List[str]:
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def get_params(self) -> Dict:
        pass


class GraphLevelEncoder(Encoder, metaclass=ABCMeta):
    @abstractmethod
    def encode(self, graph: Graph) -> GraphLatentRepresentation:
        pass

    @abstractmethod
    def loss(self, graph: Graph) -> Tuple[GraphLatentRepresentation, Dict[str, Loss]]:
        pass


class NodeLevelEncoder(Encoder, metaclass=ABCMeta):
    @abstractmethod
    def encode(self, graph: Graph) -> NodesLatentRepresentation:
        pass

    @abstractmethod
    def loss(self, graph: Graph) -> Tuple[NodesLatentRepresentation, Dict[str, Loss]]:
        pass
