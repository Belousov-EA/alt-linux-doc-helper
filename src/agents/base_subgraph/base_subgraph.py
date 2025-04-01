from abc import ABC, abstractmethod


class BaseSubgraphFabric(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_subgraph(self):
        pass
