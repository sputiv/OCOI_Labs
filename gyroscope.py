import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Gyroscope:
    """
    Класс для работы с данными гироскопа (датчика угловой скорости).
    Ведет себя как numpy массив для удобной работы с данными.
    
    Примеры использования:
        gyro = Gyroscope('data.csv', 'x')
        
        # Работа как с массивом
        print(gyro[0])           # Первый элемент
        print(gyro[10:20])       # Срез
        gyro[0] = 1.5            # Изменение значения
        
        # Использование с numpy
        filtered = np.convolve(gyro, kernel, mode='same')
        gyro_with_noise = gyro + np.random.normal(0, 0.1, len(gyro))
    """
    
    def __init__(self, data_path: str, axis: str):
        if axis.lower() not in ['x', 'y', 'z']:
            raise ValueError(f"Оси '{axis}' не существует")
        self.axis: str = axis.lower()  # Название оси

        self.csv_filepath: str = None  # Путь к данным
        self.timestamps: np.ndarray = None  # Таймстепы
        self.seconds_elapsed: np.ndarray = None  # Время прошедшее с начала измерений
        self.data: np.ndarray = None  # Текущие данные (угловая скорость)
        self.data_original: np.ndarray = None  # Исходные данные (неизменяемые)

        self._load_from_csv(data_path)

    def _load_from_csv(self, file_path: str):
        """Загружает данные из CSV файла и конвертирует в numpy массивы"""
        df = pd.read_csv(file_path)
        self.csv_filepath = file_path
        self.timestamps = df['time'].values
        self.seconds_elapsed = df['seconds_elapsed'].values
        self.data = df[self.axis].values
        self.data_original = self.data.copy()

        print(f"Загружены данные гироскопа: {len(self.timestamps)} измерений")

    def apply_noise_model(self, noise_model: 'NoiseModel') -> 'Gyroscope':
        """Применяет модель помех к данным"""
        self.data = noise_model.apply(self.data)
        return self
        
    def apply_filter(self, filter_model: 'Filter') -> 'Gyroscope':
        """Применяет фильтр к данным"""
        self.data = filter_model.apply(self.data)
        return self

    def copy(self) -> 'Gyroscope':
        """Создает глубокую копию объекта"""
        new_gyro = Gyroscope.__new__(Gyroscope)
        new_gyro.axis = self.axis
        new_gyro.csv_filepath = self.csv_filepath
        new_gyro.timestamps = self.timestamps.copy()
        new_gyro.seconds_elapsed = self.seconds_elapsed.copy()
        new_gyro.data = self.data.copy()
        new_gyro.data_original = self.data_original.copy()
        return new_gyro
    
    def reset(self) -> 'Gyroscope':
        """
        Восстанавливает данные к исходному состоянию
        
        Returns:
            self для цепочки вызовов
        """
        self.data = self.data_original.copy()
        return self
    
    def get_difference(self) -> np.ndarray:
        """
        Возвращает разницу между текущими и исходными данными
        
        Returns:
            numpy массив с разницей (data - data_original)
        """
        return self.data - self.data_original
    
    def is_modified(self) -> bool:
        """
        Проверяет, были ли данные изменены
        
        Returns:
            True если данные отличаются от исходных, False иначе
        """
        return not np.array_equal(self.data, self.data_original)

    def plot(self):
        """Отображает график данных гироскопа"""
        plt.plot(self.seconds_elapsed, self.data)
        plt.xlabel('Время (с)')
        plt.ylabel('Угловая скорость (рад/с)')
        plt.title(f'Данные гироскопа по оси {self.axis.upper()}')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def __len__(self):
        """Возвращает количество измерений"""
        return len(self.data)
    
    def __getitem__(self, index):
        """Позволяет обращаться к данным через индексы и срезы: gyro[0], gyro[10:20]"""
        return self.data[index]
    
    def __setitem__(self, index, value):
        """Позволяет изменять данные через индексы: gyro[0] = 1.5"""
        self.data[index] = value
    
    def __iter__(self):
        """Позволяет итерироваться по данным: for value in gyro"""
        return iter(self.data)
    
    def __array__(self):
        """Позволяет использовать класс в numpy функциях: np.mean(gyro)"""
        return self.data
    
    def __repr__(self):
        """Строковое представление объекта"""
        return f"Gyroscope(axis='{self.axis}', samples={len(self)}, data={self.data})"

def main():
    x = Gyroscope('data/Gyroscope_3.csv', 'x')
    y = Gyroscope('data/Gyroscope_3.csv', 'y')
    z = Gyroscope('data/Gyroscope_3.csv', 'z')

    x.plot()
    y.plot()
    z.plot()

if __name__ == '__main__':
    main()
