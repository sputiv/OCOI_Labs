# gyroscope_data.py
import numpy as np
import pandas as pd
from lab1.noise_generator import RandomNoise, NoiseModel
from lab3.filters import FilterModel
import matplotlib.pyplot as plt
import pprint

class GyroscopeData:
    def __init__(self, csv_filepath: str = None):
        self.data = None
        self.timestamps = None
        self.seconds_elapsed = None
        self.channels = None
        
        self.noised_channels = {}
        self.noise = {}
        self.noise_models = {}
        
        self.filtered_channels = {}
        self.filtered_channels_no_noise = {}
        self.filter_models = {}
        
        if csv_filepath:
            self.csv_filepath = csv_filepath
            self.load_from_csv(self.csv_filepath)
        else:
            self.csv_filepath = None
    
    def load_from_csv(self, file_path: str) -> None:
        """Загрузка данных гироскопа из CSV файла"""
        df = pd.read_csv(file_path)
        
        self.csv_filepath = file_path
        self.timestamps = df['time'].values
        self.seconds_elapsed = df['seconds_elapsed'].values
        
        self.channels = {
            'x': df['x'].values,
            'y': df['y'].values, 
            'z': df['z'].values
        }
        
        print(f"Загружены данные гироскопа: {len(self.timestamps)} измерений")
    
    def apply_noise(self, axis: str, noise_model: NoiseModel):
        """Наложение шума на данные гироскопа"""
        if axis.lower() not in ['x', 'y', 'z']:
            raise ValueError(f"Канала '{axis}' не существует")
        
        clear_signal = self.channels[axis]
        noised_signal = noise_model.apply(clear_signal)
        
        self.noised_channels[axis] = noised_signal
        self.noise_models[axis] = noise_model
        
        return noised_signal
    
    def apply_filter(self, axis: str, filter_model: FilterModel, on_noised: bool = True):
        if axis.lower() not in ['x', 'y', 'z']:
            raise ValueError(f"Канала '{axis}' не существует")

        if on_noised:
            noised_signal = self.noised_channels[axis]
            filtered_signal = filter_model.apply(noised_signal)
            
            self.filtered_channels[axis] = filtered_signal
            self.filter_models[axis] = filter_model
            
            return filtered_signal
        else:
            noised_signal = self.channels[axis]
            filtered_signal = filter_model.apply(noised_signal)
            
            self.filtered_channels_no_noise[axis] = filtered_signal
            self.filter_models[axis] = filter_model
            return filtered_signal

    def get_noise_in_axis(self, axis: str):
        """Получение шума в указанной оси"""
        if axis.lower() not in ['x', 'y', 'z']:
            raise ValueError(f"Канала '{axis}' не существует")
        
        if axis.lower() not in self.noised_channels:
            raise ValueError(f"К каналу '{axis}' не было применено модели шума")
        
        return self.noised_channels[axis] - self.channels[axis]
    
    def get_noise_model_on_axis(self, axis: str):
        """Получение модели шума для указанной оси"""
        if axis.lower() not in ['x', 'y', 'z']:
            raise ValueError(f"Канала '{axis}' не существует")
        
        if axis.lower() not in self.noised_channels:
            raise ValueError(f"К каналу '{axis}' не было применено модели шума")
        
        return self.noise_models[axis]
    
    def plot(self, use_default=True, use_noised_data=True, use_filtered_data=True):
        """Визуализация данных гироскопа"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        if not any([use_default, use_noised_data, use_filtered_data]):
            raise ValueError(f'Невозможно составить пустой график')
        
        for i, axis in enumerate(['x', 'y', 'z']):
            if use_default:
                axes[i].plot(self.seconds_elapsed, self.channels[axis], 
                            label='Исходный сигнал', linewidth=1, alpha=0.7)
            
            if use_noised_data and axis in self.noised_channels:
                axes[i].plot(self.seconds_elapsed, self.noised_channels[axis], 
                            label='С помехой', linewidth=1)
            
            if use_filtered_data and axis in self.filtered_channels:
                axes[i].plot(self.seconds_elapsed, self.filtered_channels[axis], 
                            label='Отфильтрованное', linewidth=1, alpha=0.6)
            
            axes[i].set_ylabel(f'Угловая скорость {axis} (рад/с)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        axes[2].set_xlabel('Время (сек)')
        plt.suptitle('Данные гироскопа', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def model_info(self):
        """Информация о примененных шумах"""
        print("\n" + "="*50)
        print("ИНФОРМАЦИЯ О ШУМАХ ГИРОСКОПА:")
        print("="*50)
        
        for axis in ['x', 'y', 'z']:
            try:
                noise_signal = self.get_noise_in_axis(axis)
                noise_model = self.get_noise_model_on_axis(axis)
                print(f"\nОсь {axis.upper()}:")
                print(f"  Модель шума: {type(noise_model).__name__}")
                print(f'  Параметры модели шума:')
                pprint.pprint(noise_model.__dict__)
                print(f"  Среднее значение шума: {np.mean(noise_signal):.4f}")
                print(f"  Стандартное отклонение шума: {np.std(noise_signal):.4f}")
                print(f"  Максимальное отклонение: {np.max(np.abs(noise_signal)):.4f}")
            except ValueError as e:
                print(f"\nОсь {axis.upper()}: {e}")
    
    def export_data(self, data_type: str):
        """ТИПЫ ДАННЫХ: default, filtered, filtered_no_noise, noised, noise"""
        if data_type.lower() == 'default':
            return self.channels, self.timestamps
        elif data_type.lower() == 'filtered':
            return self.filtered_channels, self.timestamps
        elif data_type.lower() == 'filtered_no_noise':
            return self.filtered_channels_no_noise, self.timestamps
        elif data_type.lower() == 'noised':
            return self.noised_channels, self.timestamps
        elif data_type.lower() == 'noise':
            return self.noise, self.timestamps