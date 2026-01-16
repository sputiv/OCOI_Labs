from abc import ABC, abstractmethod
import numpy as np

class Filter(ABC):
    """
    Абстрактный базовый класс для фильтров.
    """
    @abstractmethod
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Применяет фильтр к входным данным.
        
        Args:
            data: Входной массив данных (numpy array)
            
        Returns:
            Отфильтрованный массив данных
        """
        pass


class MovingAverage(Filter):
    """
    Реализация фильтра скользящего среднего.
    Сглаживает данные, усредняя значения в плавающем окне.
    """
    def __init__(self, window_size: int):
        """
        Инициализация фильтра.
        
        Args:
            window_size: Размер окна усреднения (количество отсчетов)
        """
        if window_size < 1:
            raise ValueError("Размер окна должен быть положительным целым числом")
        self.window_size = window_size
        
        # Создаем ядро свертки
        self.kernel = np.ones(window_size) / window_size

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Применяет скользящее среднее к данным.
        Использует свертку (convolution) с корректной обработкой краев.
        
        Вместо стандартного режима 'same' (который добавляет нули),
        мы дублируем граничные значения (padding='edge').
        Это предотвращает "линейное" искажение сигнала на границах (спад в ноль).
        """
        if len(data) < self.window_size:
            raise ValueError(f"Длина данных ({len(data)}) меньше размера окна ({self.window_size})")
            
        # Определяем размер padding
        # Для окна N нам нужно добавить (N-1) элементов суммарно
        # Centered average: N//2 слева, N//2 + (N%2 - 1) справа?
        # Проще: паддинг по (N-1)//2 слева и N//2 справа
        pad_left = (self.window_size - 1) // 2
        pad_right = self.window_size // 2
        
        # Дополняем данные значениями с краев (mode='edge')
        padded_data = np.pad(data, (pad_left, pad_right), mode='edge')
        
        # Применяем свертку в режиме 'valid' (без дополнительного паддинга нулями)
        return np.convolve(padded_data, self.kernel, mode='valid')