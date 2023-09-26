from abc import ABCMeta, abstractmethod
from numbers import Number
from typing import Dict


class AlphaMetric(metaclass=ABCMeta):

    @abstractmethod
    def evaluate(self, targets, predictions) -> Dict[str, Number]:
        pass
