# main.py - с расширенной диагностикой
from lab1.accelerometer import AccelerometerData
from lab2.giroscope import GyroscopeData
from lab2.giro_vertical import GyroVertical
from lab1.noise_generator import RandomNoise
from lab3.filters import MovingAverageFilter
import numpy as np
import matplotlib.pyplot as plt

def analyze_data_quality(accel_data, gyro_data):
    """Детальный анализ качества данных"""
    
    # Анализ акселерометра
    print("\n--- АКСЕЛЕРОМЕТР ---")
    for axis in ['x', 'y', 'z']:
        data = accel_data.channels[axis]
        
        # Основные статистики
        mean = np.mean(data)
        std = np.std(data)
        dynamic_range = np.max(data) - np.min(data)
        
        # Анализ шума (высокочастотные компоненты)
        diff = np.diff(data)
        noise_level = np.std(diff)
        
        # Отношение сигнал/шум (приблизительно)
        snr_approx = dynamic_range / (std + 1e-10)
        
        print(f"Ось {axis}:")
        print(f"  Диапазон: {dynamic_range:.3f}")
        print(f"  Стандартное отклонение: {std:.3f}")
        print(f"  Уровень шума (производная): {noise_level:.3f}")
        print(f"  Примерное SNR: {snr_approx:.2f}")
        
        # Проверка на выбросы
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        outlier_threshold = 1.5 * iqr
        outliers = np.sum((data < (q25 - outlier_threshold)) | (data > (q75 + outlier_threshold)))
        print(f"  Выбросы: {outliers} ({outliers/len(data)*100:.1f}%)")

def compare_data_versions(accel_data):
    """Сравнение разных версий данных"""
    
    # Получаем все версии данных
    default_data, _ = accel_data.export_data('default')
    noised_data, _ = accel_data.export_data('noised') 
    filtered_data, _ = accel_data.export_data('filtered')
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    
    for i, axis in enumerate(['x', 'y', 'z']):
        # Исходные данные
        axes[0, i].plot(default_data[axis], 'b-', alpha=0.7, label='Исходные')
        axes[0, i].set_title(f'Ось {axis} - Исходные')
        axes[0, i].grid(True, alpha=0.3)
        
        # Зашумленные данные
        axes[1, i].plot(noised_data[axis], 'r-', alpha=0.7, label='Зашумленные')
        axes[1, i].set_title(f'Ось {axis} - Зашумленные')
        axes[1, i].grid(True, alpha=0.3)
        
        # Отфильтрованные данные
        axes[2, i].plot(filtered_data[axis], 'g-', alpha=0.7, label='Отфильтрованные')
        axes[2, i].set_title(f'Ось {axis} - Отфильтрованные')
        axes[2, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Статистика различий
    print("\nСТАТИСТИКА РАЗЛИЧИЙ:")
    for axis in ['x', 'y', 'z']:
        orig_to_noise = np.std(noised_data[axis] - default_data[axis])
        orig_to_filtered = np.std(filtered_data[axis] - default_data[axis])
        
        print(f"Ось {axis}:")
        print(f"  Исходные → Зашумленные: σ={orig_to_noise:.4f}")
        print(f"  Исходные → Отфильтрованные: σ={orig_to_filtered:.4f}")

def test_different_noise_levels(accel_data, gyro_data):
    """Тестирование с разными уровнями шума"""
    print("\n=== ТЕСТ РАЗНЫХ УРОВНЕЙ ШУМА ===")
    
    noise_levels = [0.1, 0.5, 1.0, 2.0]
    
    fig, axes = plt.subplots(len(noise_levels), 3, figsize=(15, 4*len(noise_levels)))
    
    for i, noise_level in enumerate(noise_levels):
        # Создаем копию данных для каждого теста
        test_accel = AccelerometerData('data/accelerometer.csv')
        
        # Применяем шум
        for axis in ['x', 'y', 'z']:
            test_accel.apply_noise(axis, RandomNoise(noise_level, seed=42))
        
        # Экспортируем и обрабатываем
        accel_export = test_accel.export_data('noised')[0]
        gyro_export = gyro_data.export_data('default')[0]
        
        gyro_vertical = GyroVertical(accel_export, gyro_export, accel_data.seconds_elapsed)
        df = gyro_vertical.process()
        
        # Визуализация
        axes[i, 0].plot(df['time'], df['roll_deg'], 'r-')
        axes[i, 0].set_ylabel('Крен (°)')
        axes[i, 0].set_title(f'Шум={noise_level}')
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].plot(df['time'], df['pitch_deg'], 'g-')
        axes[i, 1].set_ylabel('Тангаж (°)')
        axes[i, 1].grid(True, alpha=0.3)
        
        axes[i, 2].plot(df['time'], df['yaw_deg'], 'b-')
        axes[i, 2].set_ylabel('Курс (°)')
        axes[i, 2].grid(True, alpha=0.3)
    
    axes[-1, 0].set_xlabel('Время (сек)')
    axes[-1, 1].set_xlabel('Время (сек)')
    axes[-1, 2].set_xlabel('Время (сек)')
    
    plt.tight_layout()
    plt.show()

def main():
  
    # Загрузка данных
    accel_data = AccelerometerData('data/accelerometer.csv')
    gyro_data = GyroscopeData('data/gyroscope.csv')
    
    # 1. Анализ исходных данных
    analyze_data_quality(accel_data, gyro_data)
    
    # 2. Применяем шум и фильтрацию
    print("\n=== ПРИМЕНЕНИЕ ШУМА И ФИЛЬТРАЦИИ ===")
    for axis in ['x', 'y', 'z']:
        accel_data.apply_noise(axis, RandomNoise(0.3, seed=42))
    
    for axis in ['x', 'y', 'z']:
        accel_data.apply_filter(axis, MovingAverageFilter(5))
    
    # 3. Сравнение версий данных
    compare_data_versions(accel_data)
    
    # 4. Тестирование разных уровней шума
    test_different_noise_levels(accel_data, gyro_data)
    
    # 5. Основные тесты гировертикали
    print("\n=== ОСНОВНЫЕ ТЕСТЫ ГИРОВЕРТИКАЛИ ===")
    
    # Тест 1: Исходные данные
    accel_export_default = accel_data.export_data('default')[0]
    gyro_export_default = gyro_data.export_data('default')[0]
    
    gyro_vertical = GyroVertical(accel_export_default, gyro_export_default, accel_data.seconds_elapsed)
    df_default = gyro_vertical.process()
    gyro_vertical.plot_orientation("Исходные данные")
    
    # Тест 2: Зашумленные данные
    accel_export_noised = accel_data.export_data('noised')[0]
    
    gyro_vertical = GyroVertical(accel_export_noised, gyro_export_default, accel_data.seconds_elapsed)
    df_noised = gyro_vertical.process()
    gyro_vertical.plot_orientation("Зашумленные данные")
    
    # Тест 3: Отфильтрованные данные
    accel_export_filtered = accel_data.export_data('filtered')[0]
    
    gyro_vertical = GyroVertical(accel_export_filtered, gyro_export_default, accel_data.seconds_elapsed)
    df_filtered = gyro_vertical.process()
    gyro_vertical.plot_orientation("Отфильтрованные данные")
    
    # Сравнительный анализ
    print("\n=== ФИНАЛЬНОЕ СРАВНЕНИЕ ===")
    compare_final_results(df_default, df_noised, df_filtered)

def compare_final_results(df_default, df_noised, df_filtered):
    """Сравнение финальных результатов"""
    metrics = ['roll_deg', 'pitch_deg', 'yaw_deg']
    
    print("\nСТАНДАРТНЫЕ ОТКЛОНЕНИЯ:")
    for metric in metrics:
        std_default = df_default[metric].std()
        std_noised = df_noised[metric].std()
        std_filtered = df_filtered[metric].std()
        
        print(f"{metric}:")
        print(f"  Исходные: {std_default:.4f}°")
        print(f"  Зашумленные: {std_noised:.4f}°")
        print(f"  Отфильтрованные: {std_filtered:.4f}°")
        
        if std_default > 0:
            change_noised = (std_noised - std_default) / std_default * 100
            change_filtered = (std_filtered - std_default) / std_default * 100
            print(f"  Изменение от исходных: шум={change_noised:+.1f}%, фильтр={change_filtered:+.1f}%")
    
    # Визуальное сравнение
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    time = df_default['time']
    
    for i, metric in enumerate(metrics):
        axes[i].plot(time, df_default[metric], 'b-', label='Исходные', alpha=0.8)
        axes[i].plot(time, df_noised[metric], 'r-', label='Зашумленные', alpha=0.6)
        axes[i].plot(time, df_filtered[metric], 'g-', label='Отфильтрованные', alpha=0.8)
        axes[i].set_ylabel(metric.replace('_deg', ' (°)'))
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Время (сек)')
    plt.suptitle('Сравнение всех версий данных')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()