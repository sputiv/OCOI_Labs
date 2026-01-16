import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from accelerometer import Accelerometer
from gyroscope import Gyroscope
from girovertical import Gyrovertical
from noise_generator import RandomNoise, SinusoidalNoise
from filter import MovingAverage

class GyroverticalExperiment:
    """
    Класс для проведения экспериментов и сравнения работы гировертикали
    с разными параметрами обработки данных.
    """
    
    def __init__(self):
        self.results: Dict[str, Gyrovertical] = {}
        self.base_sensors: Optional[Tuple] = None
        self.sensor_suffix = ""

    def load_data(self, suffix: str = "_3") -> None:
        """
        Загружает базовые данные из CSV файлов.
        
        Args:
            suffix: Суффикс файла (например, '_3' для Accelerometer_3.csv)
        """
        self.sensor_suffix = suffix
        try:
            filename_acc = f'data/Accelerometer{suffix}.csv'
            # Пробуем инициализировать, чтобы проверить наличие
            Accelerometer(filename_acc, 'x')
            print(f"Загружены данные из {filename_acc}")
        except FileNotFoundError:
            print(f"Файл {filename_acc} не найден, пробуем загрузить дефолтные")
            suffix = ""
            filename_acc = 'data/Accelerometer.csv'
            
        filename_gyro = f'data/Gyroscope{suffix}.csv'
        
        # Сохраняем "сырые" объекты, чтобы потом делать copy()
        self.base_sensors = (
            Accelerometer(filename_acc, 'x'),
            Accelerometer(filename_acc, 'y'),
            Accelerometer(filename_acc, 'z'),
            Gyroscope(filename_gyro, 'x'),
            Gyroscope(filename_gyro, 'y'),
            Gyroscope(filename_gyro, 'z')
        )
        print("Данные успешно инициализированы.")

    def _get_sensors_copy(self) -> tuple:
        """Создает независимую копию датчиков для эксперимента."""
        if self.base_sensors is None:
            raise RuntimeError("Сначала загрузите данные с помощью load_data()")
            
        return tuple(s.copy() for s in self.base_sensors)

    def _apply_filter(self, sensors: tuple, window_size: int):
        if window_size > 0:
            print(f"  Применение фильтра MovingAverage (окно {window_size})...")
            filt = MovingAverage(window_size=window_size)
            for s in sensors:
                s.apply_filter(filt)

    def add_case_random(self, name: str, amplitude: float = 0.0, filter_window: int = 0, seed: int = None) -> None:
        """
        Добавляет случай со случайным шумом (равномерное распределение).
        """
        print(f"--- Расчёт случая (Random): {name} ---")
        sensors = self._get_sensors_copy()
        
        if amplitude > 0:
            print(f"  Добавление Random шума (амплитуда {amplitude}, seed={seed})...")
            for i, s in enumerate(sensors):
                current_seed = seed + i if seed is not None else None
                s.apply_noise_model(RandomNoise(amplitude=amplitude, seed=current_seed))

        self._apply_filter(sensors, filter_window)
        
        gv = Gyrovertical(*sensors)
        gv.compute()
        self.results[name] = gv

    def add_case_sin(self, name: str, amplitude: float, frequency: float, filter_window: int = 0) -> None:
        """
        Добавляет случай с синусоидальным шумом (вибрация).
        """
        print(f"--- Расчёт случая (Синусоида): {name} ---")
        sensors = self._get_sensors_copy()
        
        # Оцениваем dt по первым двум точкам (используя seconds_elapsed для правильного масштаба времени)
        if len(sensors[0].seconds_elapsed) > 1:
            dt = sensors[0].seconds_elapsed[1] - sensors[0].seconds_elapsed[0]
        else:
            dt = 0.01
            
        print(f"  Добавление Синусоидального шума (A={amplitude}, f={frequency}Hz, dt={dt:.4f})...")
        sine_noise = SinusoidalNoise(amplitude=amplitude, frequency_hz=frequency, dt=dt)
        for s in sensors:
            s.apply_noise_model(sine_noise)

        self._apply_filter(sensors, filter_window)
        
        gv = Gyrovertical(*sensors)
        gv.compute()
        self.results[name] = gv
    
    def delete_case(self, name: str) -> None:
        if name in self.results:
            del self.results[name]

    def plot_comparison(self, names: Optional[List[str]] = None):
        """Строит графики сравнения для выбранных (или всех) случаев."""
        if not self.results:
            print("Нет результатов для отображения.")
            return

        if names:
            plot_items = {k: v for k, v in self.results.items() if k in names}
            if not plot_items:
                print(f"Ни один из запрошенных случаев {names} не найден.")
                return
        else:
            plot_items = self.results

        # Сортируем так, чтобы "Чистый" рисовался ПЕРВЫМ (снизу под остальными)
        def sort_key(item):
            name = item[0]
            if 'Clean' in name or 'ЧИСТЫЙ' in name or 'Чистый' in name:
                return 0 # В начало
            return 1 # В конец
            
        sorted_items = sorted(plot_items.items(), key=sort_key)

        # Получаем данные для оси времени
        first_gv = next(iter(plot_items.values()))
        timestamps = first_gv.timestamps
        
        default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'gray']
        default_styles = ['--', '-', ':', '-.', '-', '--', '-.', ':']

        # --- ГРАФИК 1: ДАТЧИКИ ---
        fig_sensors, axes_sensors = plt.subplots(2, 3, figsize=(18, 10))
        fig_sensors.suptitle('Сравнение искажённых данных датчиков', fontsize=16, fontweight='bold')
        
        # Заголовки столбцов
        axes_sensors[0, 0].set_title('Акселерометр X')
        axes_sensors[0, 1].set_title('Акселерометр Y')
        axes_sensors[0, 2].set_title('Акселерометр Z')
        axes_sensors[1, 0].set_title('Гироскоп X')
        axes_sensors[1, 1].set_title('Гироскоп Y')
        axes_sensors[1, 2].set_title('Гироскоп Z')

        i = 0
        for name, gv in sorted_items:
            is_clean = 'Clean' in name or 'ЧИСТЫЙ' in name or 'Чистый' in name
            
            if is_clean:
                color = 'black'
                style = '-'
                lw = 1  # Было 2.0
                alpha = 1.0 # Полностью непрозрачный
            else:
                color = default_colors[i % len(default_colors)]
                style = default_styles[i % len(default_styles)]
                lw = 0.7  # Было 1.0
                alpha = 0.75 # Полупрозрачный
                i += 1

            # Accelerometers
            axes_sensors[0, 0].plot(timestamps, gv.acc_x.data,  color=color, ls=style, lw=lw, alpha=alpha, label=name)
            axes_sensors[0, 1].plot(timestamps, gv.acc_y.data,  color=color, ls=style, lw=lw, alpha=alpha)
            axes_sensors[0, 2].plot(timestamps, gv.acc_z.data,  color=color, ls=style, lw=lw, alpha=alpha)

            # Gyroscopes
            axes_sensors[1, 0].plot(timestamps, gv.gyro_x.data, color=color, ls=style, lw=lw, alpha=alpha, label=name)
            axes_sensors[1, 1].plot(timestamps, gv.gyro_y.data, color=color, ls=style, lw=lw, alpha=alpha)
            axes_sensors[1, 2].plot(timestamps, gv.gyro_z.data, color=color, ls=style, lw=lw, alpha=alpha)

        # Легенд только в первых колонках
        axes_sensors[0, 0].legend(fontsize='small')
        axes_sensors[1, 0].legend(fontsize='small')

        for ax in axes_sensors.flat:
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Время (с)')
        
        plt.tight_layout()

        # --- ГРАФИК 2: ОРИЕНТАЦИЯ (КАК БЫЛО) ---
        fig_orient, axes_orient = plt.subplots(2, 1, figsize=(15, 12))
        
        ax_theta = axes_orient[0]
        ax_theta.set_title('Тангаж', fontsize=14, fontweight='bold')
        
        ax_gamma = axes_orient[1]
        ax_gamma.set_title('Крен', fontsize=14, fontweight='bold')
        
        i = 0
        for name, gv in sorted_items:
            is_clean = 'Clean' in name or 'ЧИСТЫЙ' in name or 'Чистый' in name
            
            if is_clean:
                color = 'black'
                style = '-'
                lw = 1.2  # Было 2.0
                alpha = 1.0
            else:
                color = default_colors[i % len(default_colors)]
                style = default_styles[i % len(default_styles)]
                lw = 0.8  # Было 1.5
                alpha = 0.6 # Полупрозрачный
                i += 1
            
            theta_deg = np.rad2deg(gv.theta)
            gamma_deg = np.rad2deg(np.unwrap(gv.gamma))
            
            ax_theta.plot(timestamps, theta_deg, label=name, color=color, linestyle=style, linewidth=lw, alpha=alpha)
            ax_gamma.plot(timestamps, gamma_deg, label=name, color=color, linestyle=style, linewidth=lw, alpha=alpha)

        ax_theta.set_ylabel('Тангаж (градусы)')
        ax_theta.grid(True, alpha=0.3)
        ax_theta.legend()
        ax_theta.axhline(y=90, color='r', linestyle=':', alpha=0.3)
        ax_theta.axhline(y=-90, color='r', linestyle=':', alpha=0.3)

        ax_gamma.set_ylabel('Крен (градусы)')
        ax_gamma.set_xlabel('Время (с)')
        ax_gamma.grid(True, alpha=0.3)
        ax_gamma.legend()
        
        plt.tight_layout()
        plt.show()


def random_noise_experiment(amplitude, seed):
    experiment_1 = GyroverticalExperiment()
    experiment_1.load_data("_3")
    exp_seed = seed
    filter_window = 2
    
    # 1. Рандомный шум
    experiment_1.add_case_random('ЧИСТЫЙ СИГНАЛ', amplitude=0, seed=exp_seed)
    experiment_1.plot_comparison()
    experiment_1.add_case_random(f'ШУМ ({amplitude})', amplitude=amplitude, seed=exp_seed)
    
    experiment_1.plot_comparison()
    experiment_1.add_case_random('Скользяшее среднее (2)', amplitude=amplitude, seed=exp_seed, filter_window=filter_window)
    experiment_1.plot_comparison()
    experiment_1.delete_case('Скользяшее среднее (2)')
    experiment_1.add_case_random('Скользяшее среднее (5)', amplitude=amplitude, seed=exp_seed, filter_window=5)
    experiment_1.plot_comparison()
    experiment_1.delete_case('Скользяшее среднее (5)')
    experiment_1.add_case_random('Скользяшее среднее (10)', amplitude=amplitude, seed=exp_seed, filter_window=10)
    experiment_1.plot_comparison()

def sin_noise_experiment(amplitude, frequency):
    experiment_2 = GyroverticalExperiment()
    experiment_2.load_data("_3")
    exp_seed = 77
    sin_amplitude = amplitude
    sin_freq = frequency

    experiment_2.add_case_random('ЧИСТЫЙ СИГНАЛ', amplitude=0, seed=exp_seed)
    experiment_2.add_case_sin(f'СИНУСОИДАЛЬНЫЙ ШУМ ({sin_amplitude}, {sin_freq} Гц)', amplitude=sin_amplitude, frequency=sin_freq, filter_window=0)
    experiment_2.plot_comparison()

    experiment_2.add_case_sin('Скользяшее среднее (2)', amplitude=sin_amplitude, frequency=sin_freq, filter_window=2)
    experiment_2.plot_comparison()
    experiment_2.delete_case('Скользяшее среднее (2)')
    experiment_2.add_case_sin('Скользяшее среднее (5)', amplitude=sin_amplitude, frequency=sin_freq, filter_window=5)
    experiment_2.plot_comparison()
    experiment_2.delete_case('Скользяшее среднее (5)')
    experiment_2.add_case_sin('Скользяшее среднее (10)', amplitude=sin_amplitude, frequency=sin_freq, filter_window=10)
    experiment_2.plot_comparison()

def clear_input_experiment():
    experiment = GyroverticalExperiment()
    experiment.load_data("_3")
    exp_seed = 77
    experiment.add_case_random('ВХОДНОЙ СИГНАЛ', amplitude=0, seed=exp_seed)
    experiment.add_case_random('Скользяшее среднее (2)', amplitude=0, seed=exp_seed, filter_window=2)
    experiment.plot_comparison()
    experiment.delete_case('Скользяшее среднее (2)')
    experiment.add_case_random('Скользяшее среднее (5)', amplitude=0, seed=exp_seed, filter_window=5)
    experiment.plot_comparison()
    experiment.delete_case('Скользяшее среднее (5)')
    experiment.add_case_random('Скользяшее среднее (10)', amplitude=0, seed=exp_seed, filter_window=10)
    experiment.plot_comparison()


def main():

    # Эксперимент с рандомным шумом (амплитуда 0.5, сид 43)
    random_noise_experiment(0.5, 43)

    # Эксперимент с синусоидальным шумом (амплитуда 0.3, 5 Гц)
    sin_noise_experiment(0.3, 5)

    # Эксперимент улучшения качества исходного сигнала
    clear_input_experiment()

if __name__ == '__main__':
    main()
