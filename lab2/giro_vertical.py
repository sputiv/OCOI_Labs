# gyro_vertical.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class GyroVertical:
    """Алгоритм гировертикали - универсальная версия для работы с сырыми данными"""
    
    def __init__(self, accel_data: dict, gyro_data: dict, timestamps: np.ndarray, epsilon=0.0003, g=9.8):
        """
        Универсальный конструктор для работы с любыми данными
        
        Args:
            accel_data: словарь с данными акселерометра {'x': np.array, 'y': np.array, 'z': np.array}
            gyro_data: словарь с данными гироскопа {'x': np.array, 'y': np.array, 'z': np.array}  
            timestamps: массив временных меток в секундах
            epsilon: коэффициент крутизны коррекции (увеличено для стабильности)
            g: ускорение свободного падения
        """
        self.accel_data = accel_data
        self.gyro_data = gyro_data
        self.timestamps = timestamps
        self.epsilon = epsilon
        self.g = g
        
        self._validate_data()
        
        # Матрица ориентации
        self.C = np.eye(3)
        
        # Результаты
        self.orientations = []
    
    def _validate_data(self):
        """Проверка корректности входных данных"""
        required_axes = ['x', 'y', 'z']
        
        for axis in required_axes:
            if axis not in self.accel_data:
                raise ValueError(f"Отсутствуют данные акселерометра по оси {axis}")
            if axis not in self.gyro_data:
                raise ValueError(f"Отсутствуют данные гироскопа по оси {axis}")
        
        # Проверяем одинаковую длину данных
        accel_lengths = [len(self.accel_data[axis]) for axis in required_axes]
        gyro_lengths = [len(self.gyro_data[axis]) for axis in required_axes]
        
        if len(set(accel_lengths)) != 1:
            raise ValueError("Данные акселерометра по осям имеют разную длину")
        if len(set(gyro_lengths)) != 1:
            raise ValueError("Данные гироскопа по осям имеют разную длину")
        if accel_lengths[0] != gyro_lengths[0]:
            raise ValueError("Данные акселерометра и гироскопа имеют разную длину")
        if accel_lengths[0] != len(self.timestamps):
            raise ValueError("Длина временных меток не соответствует длине данных")
    
    @classmethod
    def from_sensor_objects(cls, accelerometer_data, gyroscope_data, epsilon=0.003, g=9.8):
        """
        Альтернативный конструктор для работы с объектами AccelerometerData и GyroscopeData
        """
        # Извлекаем данные из объектов
        accel_data = {
            'x': accelerometer_data.channels['x'],
            'y': accelerometer_data.channels['y'], 
            'z': accelerometer_data.channels['z']
        }
        
        gyro_data = {
            'x': gyroscope_data.channels['x'],
            'y': gyroscope_data.channels['y'],
            'z': gyroscope_data.channels['z']
        }
        
        timestamps = accelerometer_data.seconds_elapsed
        
        return cls(accel_data, gyro_data, timestamps, epsilon, g)
    
    def initialize(self):
        """Инициализация по первому измерению акселерометра"""
        accel_x = self.accel_data['x'][0]
        accel_y = self.accel_data['y'][0]
        accel_z = self.accel_data['z'][0]
        
        # Вычисляем начальные углы по акселерометру
        pitch = np.arctan2(-accel_x, np.sqrt(accel_y**2 + accel_z**2))
        roll = np.arctan2(accel_y, accel_z)
        
        # Строим начальную матрицу ориентации (проверенная версия)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        
        self.C = np.array([
            [cp, sp*sr, sp*cr],
            [0, cr, -sr],
            [-sp, cp*sr, cp*cr]
        ])
        
        print(f"Инициализация завершена: крен={np.degrees(roll):.2f}°, тангаж={np.degrees(pitch):.2f}°")
    
    def _skew_symmetric(self, vector):
        """Создание кососимметрической матрицы из вектора"""
        return np.array([
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0]
        ])
    
    def _orthogonalize_matrix(self, C):
        """Ортогонализация матрицы методом Грама-Шмидта"""
        c0 = C[:, 0] / np.linalg.norm(C[:, 0])
        c1 = C[:, 1] - np.dot(C[:, 1], c0) * c0
        c1 = c1 / np.linalg.norm(c1)
        c2 = np.cross(c0, c1)
        return np.column_stack([c0, c1, c2])
    
    def _get_orientation_angles(self):
        """Извлечение углов Эйлера из матрицы ориентации (стабильная версия)"""
        # Ограничиваем значение для арксинуса
        sin_pitch = -self.C[2, 0]
        sin_pitch = np.clip(sin_pitch, -0.999999, 0.999999)
        pitch = np.arcsin(sin_pitch)
        
        # Вычисляем крен и курс с проверкой на сингулярности
        if np.abs(sin_pitch) < 0.999999:
            roll = np.arctan2(self.C[2, 1], self.C[2, 2])
            yaw = np.arctan2(self.C[1, 0], self.C[0, 0])
        else:
            # Вблизи сингулярности (тангаж ±90°)
            roll = 0
            yaw = np.arctan2(-self.C[0, 1], self.C[1, 1])
        
        return roll, pitch, yaw
    
    def process(self):
        """Обработка всех данных алгоритмом гировертикали"""
        self.initialize()
        self.orientations = []
        
        n_points = len(self.timestamps)
        
        for i in range(n_points):
            # Получаем данные для текущего шага
            accel = np.array([
                self.accel_data['x'][i],
                self.accel_data['y'][i],
                self.accel_data['z'][i]
            ])
            
            gyro = np.array([
                self.gyro_data['x'][i],
                self.gyro_data['y'][i],
                self.gyro_data['z'][i]
            ])
            
            # Вычисляем шаг времени
            if i == 0:
                dt = 0.01  # минимальный шаг по умолчанию
            else:
                dt = self.timestamps[i] - self.timestamps[i-1]
                if dt <= 0:
                    dt = 0.01  # защита от некорректных шагов
            
            if dt > 0:  # Пропускаем нулевые шаги
                # Нормализуем ускорение
                accel_norm = np.linalg.norm(accel)
                if accel_norm == 0:
                    n_xyz = np.array([0, 0, 1])
                else:
                    n_xyz = accel / accel_norm
                
                # Основной алгоритм гировертикали (рабочая версия)
                n_ENH = self.C @ n_xyz
                
                # Коррекция только по горизонтальным осям
                omega_EN = np.array([
                    -self.epsilon * n_ENH[1],  # Восточная компонента
                    self.epsilon * n_ENH[0],   # Северная компонента  
                    0                          # Вертикальная - свободная в азимуте
                ])
                
                # Матрицы угловых скоростей
                omega_gyro_skew = self._skew_symmetric(gyro)
                omega_EN_skew = self._skew_symmetric(omega_EN)
                
                # Обновление матрицы ориентации (проверенное уравнение)
                dCdt = omega_EN_skew @ self.C - self.C @ omega_gyro_skew
                self.C = self.C + dt * dCdt
                
                # Ортогонализация матрицы
                self.C = self._orthogonalize_matrix(self.C)
            
            # Сохраняем ориентацию
            roll, pitch, yaw = self._get_orientation_angles()
            self.orientations.append({
                'time': self.timestamps[i],
                'roll': roll,
                'pitch': pitch,
                'yaw': yaw,
                'roll_deg': np.degrees(roll),
                'pitch_deg': np.degrees(pitch),
                'yaw_deg': np.degrees(yaw)
            })
        
        return pd.DataFrame(self.orientations)
    
    def plot_orientation(self, title="Ориентация по алгоритму гировертикали"):
        """Визуализация результатов ориентации"""
        if not self.orientations:
            raise ValueError("Сначала выполните process()")
        
        df = pd.DataFrame(self.orientations)
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        axes[0].plot(df['time'], df['roll_deg'], label='Крен (Roll)', linewidth=2, color='red')
        axes[0].set_ylabel('Крен (°)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(df['time'], df['pitch_deg'], label='Тангаж (Pitch)', linewidth=2, color='green')
        axes[1].set_ylabel('Тангаж (°)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(df['time'], df['yaw_deg'], label='Курс (Yaw)', linewidth=2, color='blue')
        axes[2].set_ylabel('Курс (°)')
        axes[2].set_xlabel('Время (сек)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def get_orientation_df(self):
        """Получение результатов в виде DataFrame"""
        if not self.orientations:
            raise ValueError("Сначала выполните process()")
        return pd.DataFrame(self.orientations)
    
    def get_final_orientation(self):
        """Получение финальной ориентации"""
        if not self.orientations:
            return None
        return self.orientations[-1]