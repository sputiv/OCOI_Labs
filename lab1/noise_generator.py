from abc import ABC, abstractmethod
import numpy as np
import random

class NoiseModel(ABC):
    """Абстрактный базовый класс для моделей помех"""
    
    @abstractmethod
    def apply(self, clean_data: np.ndarray) -> np.ndarray:
        pass

class RandomNoise(NoiseModel):
    def __init__(self, amplitude: float, seed: int = None):
        self.amplitude: float = amplitude
        self.seed = seed if seed else random.randint(1, 1000)
        self.rng = np.random.RandomState(seed)
    
    def apply(self, clean_data: np.ndarray) -> np.ndarray:
        noise = self.rng.uniform(-self.amplitude, self.amplitude, clean_data.shape)
        return clean_data + noise
    
    def __repr__(self):
        return f"RandomNoise(amplitude={self.amplitude}, seed={self.seed})"