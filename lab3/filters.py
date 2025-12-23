# filters.py
import numpy as np
from typing import Union, List
from abc import ABC, abstractmethod

class FilterModel(ABC):
    @abstractmethod
    def apply(self, data: Union[List[float], np.ndarray]) -> np.ndarray:
        pass

class MovingAverageFilter(FilterModel):
    def __init__(self, window_size: int = 5):
        if window_size < 1:
            raise ValueError("Размер окна должен быть положительным")
        self.window_size = window_size
    
    def apply(self, data: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        ИСПРАВЛЕННАЯ РЕАЛИЗАЦИЯ
        Использует правильный алгоритм скользящего среднего
        """
        data = np.array(data)
        if len(data) < self.window_size:
            return data
        
        # Правильная реализация через cumsum
        cumsum = np.cumsum(np.insert(data, 0, 0))
        filtered = (cumsum[self.window_size:] - cumsum[:-self.window_size]) / float(self.window_size)
        
        # Дополняем начало массива
        prefix = []
        for i in range(1, self.window_size):
            prefix.append(np.mean(data[:i]))
        
        result = np.concatenate([prefix, filtered])
        return result
    
    def __repr__(self):
        return f"MovingAverageFilter(window_size={self.window_size})"