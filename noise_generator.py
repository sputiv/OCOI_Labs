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

class SinusoidalNoise(NoiseModel):
    """
    Генерирует синусоидальную помеху (например, вибрация двигателя).
    noise = A * sin(2*pi*f*t + phi)
    """
    def __init__(self, amplitude: float, frequency_hz: float, phase_rad: float = 0.0, dt: float = 0.01):
        """
        Args:
            amplitude: Амплитуда помехи
            frequency_hz: Частота помехи в Гц
            phase_rad: Начальная фаза в радианах
            dt: Шаг дискретизации по времени (сек), по умолчанию 0.01 (100 Гц)
        """
        self.amplitude = amplitude
        self.frequency = frequency_hz
        self.phase = phase_rad
        self.dt = dt
        
    def apply(self, clean_data: np.ndarray) -> np.ndarray:
        n_samples = len(clean_data)
        t = np.arange(n_samples) * self.dt
        noise = self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)
        return clean_data + noise

    def __repr__(self):
        return f"SinusoidalNoise(A={self.amplitude}, f={self.frequency}Hz, dt={self.dt})"