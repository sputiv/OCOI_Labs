import numpy as np
import matplotlib.pyplot as plt
from gyroscope import Gyroscope
from noise_generator import RandomNoise
from filter import MovingAverage

def main():
    # 1. Загружаем и копируем гироскоп
    # Используем дефолтный если _3 нет, но скрипты выше использовали _3
    try:
        gyro = Gyroscope('data/Gyroscope_3.csv', 'z')
    except:
        gyro = Gyroscope('data/Gyroscope.csv', 'z')
        
    # Берем срез данных (500 точек) для наглядности
    raw_data = gyro.data[:500].copy()
    
    # 2. Добавляем шум
    noise = RandomNoise(amplitude=0.5, seed=42)
    noisy_data = noise.apply(raw_data.copy())
    
    # 3. Фильтруем разными окнами
    ma2 = MovingAverage(2)
    ma5 = MovingAverage(5)
    ma10 = MovingAverage(10)
    
    filtered_2 = ma2.apply(noisy_data.copy())
    filtered_5 = ma5.apply(noisy_data.copy())
    filtered_10 = ma10.apply(noisy_data.copy())
    
    # 4. Считаем статистику (STD шума)
    # Чтобы оценить шум, вычтем "чистый" сигнал (хотя он тоже меняется, но грубо)
    # Лучше просто покажем графики
    
    print(f"STD Noisy: {np.std(noisy_data - raw_data):.4f}")
    print(f"STD Filtered 2: {np.std(filtered_2 - raw_data):.4f}")
    print(f"STD Filtered 5: {np.std(filtered_5 - raw_data):.4f}")
    print(f"STD Filtered 10: {np.std(filtered_10 - raw_data):.4f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(noisy_data, color='gray', alpha=0.5, label='Noisy (0.5)')
    plt.plot(filtered_2, color='blue', alpha=0.8, label='MA(2)')
    plt.plot(filtered_10, color='red', linewidth=2, label='MA(10)')
    plt.plot(raw_data, color='black', linestyle='--', label='Clean')
    plt.legend()
    plt.title("Проверка фильтрации на сырых данных гироскопа (Z)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
