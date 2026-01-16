"""
Реализация алгоритма гировертикали для расчёта параметров ориентации.

Класс Gyrovertical вычисляет углы ориентации (курс, тангаж, крен) на основе
данных акселерометров и гироскопов с использованием метода Рунге-Кутта 4-го порядка.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings

from accelerometer import Accelerometer
from gyroscope import Gyroscope


class Gyrovertical:
    """
    Алгоритм гировертикали для определения параметров ориентации объекта.
    
    Использует данные 3-х акселерометров и 3-х гироскопов для расчёта:
    - ψ (psi) - угол курса (yaw)
    - υ (theta) - угол тангажа (pitch)
    - γ (gamma) - угол крена (roll)
    
    Примеры использования:
        acc_x = Accelerometer('data.csv', 'x')
        acc_y = Accelerometer('data.csv', 'y')
        acc_z = Accelerometer('data.csv', 'z')
        gyro_x = Gyroscope('data.csv', 'x')
        gyro_y = Gyroscope('data.csv', 'y')
        gyro_z = Gyroscope('data.csv', 'z')
        
        gv = Gyrovertical(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
        gv.compute()
        gv.plot_angles()
    """
    
    # Физические константы
    G_E = 9.78049  # Ускорение силы тяжести на экваторе, м/с²
    Q = 0.00346775  # Отношение центробежной силы к силе тяжести на экваторе
    BETA = 0.0053171
    BETA1 = 71e-7
    
    # Ограничения
    MAX_PITCH_DEG = 150.0  # Максимальный допустимый тангаж в градусах
    MAX_PITCH_RAD = np.deg2rad(MAX_PITCH_DEG)
    
    def __init__(
        self,
        acc_x: Accelerometer,
        acc_y: Accelerometer,
        acc_z: Accelerometer,
        gyro_x: Gyroscope,
        gyro_y: Gyroscope,
        gyro_z: Gyroscope,
        epsilon: float = 1.1542e-4,  # Коэффициент крутизны коррекции
        psi_0: float = 0.0,          # Начальный курс (рад)
        theta_0: float = 0.0,        # Начальный тангаж (рад)
        gamma_0: float = 0.0,        # Начальный крен (рад)
        latitude: float = 0.0,       # Широта φ (рад)
        altitude: float = 0.0        # Высота h (м)
    ):
        """
        Инициализация алгоритма гировертикали.
        
        Args:
            acc_x, acc_y, acc_z: Объекты Accelerometer для осей X, Y, Z
            gyro_x, gyro_y, gyro_z: Объекты Gyroscope для осей X, Y, Z
            epsilon: Коэффициент крутизны коррекции
            psi_0: Начальный угол курса в радианах
            theta_0: Начальный угол тангажа в радианах
            gamma_0: Начальный угол крена в радианах
            latitude: Широта в радианах
            altitude: Высота над уровнем моря в метрах
        """
        # Сохранение датчиков
        self.acc_x = acc_x
        self.acc_y = acc_y
        self.acc_z = acc_z
        self.gyro_x = gyro_x
        self.gyro_y = gyro_y
        self.gyro_z = gyro_z
        
        # Параметры алгоритма
        self.epsilon = epsilon
        self.latitude = latitude
        self.altitude = altitude
        
        # Начальные углы (нормализованные)
        self.psi_0 = self._normalize_angle(psi_0)
        self.theta_0 = self._normalize_angle(theta_0)
        self.gamma_0 = self._normalize_angle(gamma_0)
        
        # Проверка начального тангажа
        if not self._validate_pitch(self.theta_0):
            warnings.warn(
                f"Начальный тангаж {np.rad2deg(self.theta_0):.1f}° превышает "
                f"допустимый предел ±{self.MAX_PITCH_DEG}°. "
                f"Гировертикаль может работать некорректно!"
            )
        
        # Расчёт силы тяжести
        self.g_z = self._calculate_gravity()
        
        # Инициализация матрицы ориентации
        self.C_0 = self._initialize_orientation_matrix()
        
        # Результаты (будут заполнены после compute())
        self.timestamps: Optional[np.ndarray] = None
        self.C_history: Optional[np.ndarray] = None  # История матриц ориентации
        self.psi: Optional[np.ndarray] = None        # История углов курса
        self.theta: Optional[np.ndarray] = None      # История углов тангажа
        self.gamma: Optional[np.ndarray] = None      # История углов крена
        
        print(f"Инициализирована гировертикаль:")
        print(f"  Начальные углы: ψ={np.rad2deg(self.psi_0):.2f}°, "
              f"υ={np.rad2deg(self.theta_0):.2f}°, γ={np.rad2deg(self.gamma_0):.2f}°")
        print(f"  Коэффициент коррекции ε={self.epsilon:.6e}")
        print(f"  Сила тяжести g_z={self.g_z:.4f} м/с²")
    
    def _normalize_angle(self, angle: float) -> float:
        """
        Нормализует угол в диапазон [-π, π].
        
        Использует atan2 для корректной обработки всех квадрантов
        и предотвращения разрывов на графиках.
        
        Args:
            angle: Угол в радианах
            
        Returns:
            Нормализованный угол в диапазоне [-π, π]
        """
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def _validate_pitch(self, theta: float) -> bool:
        """
        Проверяет что тангаж находится в допустимых пределах ±150°.
        
        Args:
            theta: Угол тангажа в радианах
            
        Returns:
            True если тангаж в допустимых пределах, False иначе
        """
        return abs(theta) <= self.MAX_PITCH_RAD
    
    def _calculate_gravity(self) -> float:
        """
        Рассчитывает модуль силы тяжести в текущей точке Земли.
        
        Формула из документации (строки 289-299):
        g_z = g_0 + (h/a) * (...)
        
        Для упрощения при latitude=0 и altitude=0 возвращает g_e.
        
        Returns:
            Модуль силы тяжести в м/с²
        """
        phi = self.latitude
        h = self.altitude
        
        # Расчёт g_0
        g_0 = self.G_E * (1 + self.BETA * np.sin(phi)**2 + self.BETA1 * np.sin(2*phi)**2)
        
        # Для упрощения при h=0 возвращаем g_0
        if h == 0:
            return g_0
        
        # Полная формула (если нужна в будущем)
        # a = 6378137  # Большая полуось Земли, м
        # e2 = 0.00669438  # Квадрат эксцентриситета
        # g_z = g_0 + (h/a) * (3*h/a - 2*self.Q*self.G_E*np.cos(phi)**2 + 
        #                       e2*(3*np.sin(phi)**2 - 1) - 
        #                       self.Q*(1 + 6*np.sin(phi)**2))
        
        return g_0
    
    def _initialize_orientation_matrix(self) -> np.ndarray:
        """
        Создаёт начальную матрицу ориентации C[0] из начальных углов.
        
        Формула из документации (строки 232-267), формула (28).
        
        Returns:
            Матрица ориентации 3x3
        """
        psi = self.psi_0
        theta = self.theta_0
        gamma = self.gamma_0
        
        C = np.zeros((3, 3))
        
        # Строка 0
        C[0, 0] = -np.sin(psi) * np.cos(theta)
        C[0, 1] = np.sin(psi) * np.sin(theta) * np.cos(gamma) + np.cos(psi) * np.sin(gamma)
        C[0, 2] = -np.sin(psi) * np.sin(theta) * np.sin(gamma) + np.cos(psi) * np.cos(gamma)
        
        # Строка 1
        C[1, 0] = np.cos(psi) * np.cos(theta)
        C[1, 1] = -np.cos(psi) * np.sin(theta) * np.cos(gamma) + np.sin(psi) * np.sin(gamma)
        C[1, 2] = np.cos(psi) * np.sin(theta) * np.sin(gamma) + np.sin(psi) * np.cos(gamma)
        
        # Строка 2
        C[2, 0] = np.sin(theta)
        C[2, 1] = np.cos(theta) * np.cos(gamma)
        C[2, 2] = -np.cos(theta) * np.sin(gamma)
        
        return C
    
    def _create_skew_symmetric_matrix(self, omega: np.ndarray) -> np.ndarray:
        """
        Создаёт кососимметричную матрицу из вектора угловой скорости.
        
        Для вектора ω = [ω_x, ω_y, ω_z] создаёт матрицу:
        [Ω] = [    0   -ω_z    ω_y ]
              [  ω_z     0   -ω_x ]
              [ -ω_y   ω_x      0 ]
        
        Args:
            omega: Вектор угловой скорости [ω_x, ω_y, ω_z]
            
        Returns:
            Кососимметричная матрица 3x3
        """
        return np.array([
            [0,        -omega[2],  omega[1]],
            [omega[2],  0,        -omega[0]],
            [-omega[1], omega[0],  0       ]
        ])
    
    def _calculate_dC_dt(
        self,
        C: np.ndarray,
        omega_hor: np.ndarray,
        omega_gyr: np.ndarray
    ) -> np.ndarray:
        """
        Рассчитывает производную матрицы ориентации dC/dt.
        
        Использует модифицированное уравнение Пуассона (строки 70-86):
        dC/dt = [Ω_гор] × C + C × [Ω_гир]
        
        где Ω_гор = [Ω_E, Ω_N, Ω_H] - угловые скорости горизонтной СК
            Ω_гир = [Ω_1, Ω_2, Ω_3] - угловые скорости связанной СК (от гироскопов)
        
        Args:
            C: Текущая матрица ориентации 3x3
            omega_hor: Вектор угловых скоростей горизонтной СК [Ω_E, Ω_N, Ω_H]
            omega_gyr: Вектор угловых скоростей связанной СК [Ω_1, Ω_2, Ω_3]
            
        Returns:
            Производная матрицы ориентации dC/dt
        """
        # Создаём кососимметричные матрицы согласно формуле из документации
        # Для Ω_гор (строки 73-75): используем [Ω_x, Ω_y, Ω_z] = [Ω_E, Ω_N, Ω_H]
        Omega_hor = self._create_skew_symmetric_matrix(omega_hor)
        
        # Для Ω_гир (строки 82-84): используем [Ω_1, Ω_2, Ω_3]
        Omega_gyr = self._create_skew_symmetric_matrix(omega_gyr)
        
        # Модифицированное уравнение Пуассона (строки 71-85)
        # dC/dt = [Ω_гор] @ C + C @ [Ω_гир]
        dC_dt = Omega_hor @ C + C @ Omega_gyr
        
        return dC_dt
    
    def _extract_angles(self, C: np.ndarray) -> Tuple[float, float, float]:
        """
        Извлекает углы ориентации из матрицы ориентации.
        
        Формулы из документации (строки 481-497).
        
        Args:
            C: Матрица ориентации 3x3
            
        Returns:
            Кортеж (psi, theta, gamma) - углы в радианах, нормализованные в [-π, π]
        """
        sqrt2_2 = np.sqrt(2) / 2
        
        # Угол курса ψ (строки 483-485)
        if abs(C[1, 0]) > sqrt2_2:
            psi = np.arctan2(-C[0, 0], C[1, 0])
        else:
            psi = np.pi/2 - np.arctan2(C[1, 0], -C[0, 0])
        
        # Угол тангажа υ (строки 489-490)
        # Формула из документации: θ = arcsin(C[2,0])
        theta = np.arcsin(np.clip(C[2, 0], -1.0, 1.0))
        
        # Угол крена γ (строки 495-497)
        if abs(C[2, 1]) > sqrt2_2:
            gamma = np.arctan2(-C[2, 2], C[2, 1])
        else:
            gamma = np.pi/2 - np.arctan2(C[2, 1], -C[2, 2])
        
        # Нормализация всех углов в диапазон [-π, π]
        psi = self._normalize_angle(psi)
        theta = self._normalize_angle(theta)
        gamma = self._normalize_angle(gamma)
        
        # Проверка тангажа
        if not self._validate_pitch(theta):
            warnings.warn(
                f"Тангаж {np.rad2deg(theta):.1f}° превышает допустимый предел "
                f"±{self.MAX_PITCH_DEG}°. Гировертикаль может работать некорректно!",
                stacklevel=2
            )
        
        return psi, theta, gamma
    
    def compute(self) -> None:
        """
        Выполняет расчёт параметров ориентации для всех измерений.
        
        Использует метод Рунге-Кутта 4-го порядка для численного интегрирования
        матрицы ориентации. На каждом шаге извлекает углы ориентации.
        
        Результаты сохраняются в атрибутах:
        - self.timestamps: временные метки
        - self.C_history: история матриц ориентации
        - self.psi: углы курса
        - self.theta: углы тангажа
        - self.gamma: углы крена
        """
        print("\nЗапуск расчёта гировертикали...")
        
        # Проверка что все датчики имеют одинаковую длину
        n_samples = len(self.acc_x)
        if not all(len(sensor) == n_samples for sensor in 
                   [self.acc_y, self.acc_z, self.gyro_x, self.gyro_y, self.gyro_z]):
            raise ValueError("Все датчики должны иметь одинаковое количество измерений")
        
        # Инициализация массивов для результатов
        self.timestamps = self.acc_x.seconds_elapsed
        self.C_history = np.zeros((n_samples, 3, 3))
        self.psi = np.zeros(n_samples)
        self.theta = np.zeros(n_samples)
        self.gamma = np.zeros(n_samples)
        
        # Начальные условия
        C_prev = self.C_0.copy()
        self.C_history[0] = C_prev
        self.psi[0], self.theta[0], self.gamma[0] = self._extract_angles(C_prev)
        
        print(f"Обработка {n_samples} измерений...")
        
        # Основной цикл интегрирования
        for i in range(1, n_samples):
            # Шаг интегрирования
            dt = self.timestamps[i] - self.timestamps[i-1]
            
            # Данные акселерометров и гироскопов на текущем шаге
            # Для Рунге-Кутта нужны 4 точки (k=0,1,2,3)
            # Упрощение: используем одни и те же данные для всех k
            acc_data = np.array([self.acc_x[i], self.acc_y[i], self.acc_z[i]])
            gyro_data = np.array([self.gyro_x[i], self.gyro_y[i], self.gyro_z[i]])
            
            # Рунге-Кутта 4-го порядка
            # k0: начало шага
            C_k0 = C_prev
            n_hor_k0 = C_k0 @ (self.g_z * acc_data)  # Пересчёт в горизонтную СК
            omega_hor_k0 = np.array([-self.epsilon * n_hor_k0[1],  # Ω_E
                                      self.epsilon * n_hor_k0[0],   # Ω_N
                                      0.0])                          # Ω_H
            dC_k0 = self._calculate_dC_dt(C_k0, omega_hor_k0, gyro_data)
            
            # k1: середина шага (первая оценка)
            C_k1 = C_prev + 0.5 * dt * dC_k0
            n_hor_k1 = C_k1 @ (self.g_z * acc_data)
            omega_hor_k1 = np.array([-self.epsilon * n_hor_k1[1],
                                      self.epsilon * n_hor_k1[0],
                                      0.0])
            dC_k1 = self._calculate_dC_dt(C_k1, omega_hor_k1, gyro_data)
            
            # k2: середина шага (вторая оценка)
            C_k2 = C_prev + 0.5 * dt * dC_k1
            n_hor_k2 = C_k2 @ (self.g_z * acc_data)
            omega_hor_k2 = np.array([-self.epsilon * n_hor_k2[1],
                                      self.epsilon * n_hor_k2[0],
                                      0.0])
            dC_k2 = self._calculate_dC_dt(C_k2, omega_hor_k2, gyro_data)
            
            # k3: конец шага
            C_k3 = C_prev + dt * dC_k2
            n_hor_k3 = C_k3 @ (self.g_z * acc_data)
            omega_hor_k3 = np.array([-self.epsilon * n_hor_k3[1],
                                      self.epsilon * n_hor_k3[0],
                                      0.0])
            dC_k3 = self._calculate_dC_dt(C_k3, omega_hor_k3, gyro_data)
            
            # Формула Рунге-Кутта 4-го порядка (строка 478)
            C_new = C_prev + (dt / 6.0) * (dC_k0 + 2*dC_k1 + 2*dC_k2 + dC_k3)
            
            # Сохранение результатов
            self.C_history[i] = C_new
            self.psi[i], self.theta[i], self.gamma[i] = self._extract_angles(C_new)
            
            # Обновление для следующей итерации
            C_prev = C_new
            
            # Прогресс
            if (i % 200 == 0) or (i == n_samples - 1):
                progress = 100 * i / (n_samples - 1)
                print(f"  Прогресс: {progress:.1f}% ({i}/{n_samples})")
        
        print(f"✓ Расчёт завершён!")
        print(f"  Диапазон углов:")
        print(f"    Курс (ψ):   [{np.rad2deg(self.psi.min()):.1f}°, {np.rad2deg(self.psi.max()):.1f}°]")
        print(f"    Тангаж (υ): [{np.rad2deg(self.theta.min()):.1f}°, {np.rad2deg(self.theta.max()):.1f}°]")
        print(f"    Крен (γ):   [{np.rad2deg(self.gamma.min()):.1f}°, {np.rad2deg(self.gamma.max()):.1f}°]")
    
    def __repr__(self):
        return (f"Gyrovertical(epsilon={self.epsilon:.6e}, "
                f"initial_angles=(ψ={np.rad2deg(self.psi_0):.1f}°, "
                f"υ={np.rad2deg(self.theta_0):.1f}°, "
                f"γ={np.rad2deg(self.gamma_0):.1f}°))")
    
    def plot_angles(self, in_degrees: bool = True, unwrap: bool = True) -> None:
        """
        Отображает графики углов ориентации во времени.
        
        Args:
            in_degrees: Если True, отображает углы в градусах, иначе в радианах
            unwrap: Если True, устраняет разрывы при переходе через ±π (рекомендуется)
                   Примечание: unwrap НЕ применяется к тангажу, так как он ограничен ±90°
        """
        if self.psi is None:
            raise RuntimeError("Сначала необходимо вызвать метод compute()")
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Углы ориентации', fontsize=16, fontweight='bold')
        
        # Устранение разрывов при переходе через ±π (КРИТИЧЕСКИ ВАЖНО!)
        # ВАЖНО: unwrap НЕ применяется к тангажу, так как он ограничен диапазоном ±90°
        if unwrap:
            psi_plot = np.unwrap(self.psi)
            theta_plot = self.theta.copy()  # Тангаж НЕ unwrap!
            gamma_plot = np.unwrap(self.gamma)
        else:
            psi_plot = self.psi.copy()
            theta_plot = self.theta.copy()
            gamma_plot = self.gamma.copy()
        
        # Конвертация в градусы если нужно
        if in_degrees:
            psi_plot = np.rad2deg(psi_plot)
            theta_plot = np.rad2deg(theta_plot)
            gamma_plot = np.rad2deg(gamma_plot)
            unit = '°'
        else:
            unit = 'рад'
        
        # График курса
        axes[0].plot(self.timestamps, psi_plot, 'b-', linewidth=1.5, label='Курс (ψ)')
        axes[0].set_ylabel(f'Курс (ψ), {unit}', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper right')
        if unwrap:
            axes[0].set_title('Курс (с устранением разрывов)', fontsize=10, style='italic')
        
        # График тангажа
        axes[1].plot(self.timestamps, theta_plot, 'r-', linewidth=1.5, label='Тангаж (υ)')
        # Показываем пределы ±90° (физический предел для arcsin)
        if in_degrees:
            axes[1].axhline(y=90, color='red', linestyle=':', alpha=0.5, linewidth=1, label='Предел ±90°')
            axes[1].axhline(y=-90, color='red', linestyle=':', alpha=0.5, linewidth=1)
            # Показываем предел гировертикали ±150°
            axes[1].axhline(y=self.MAX_PITCH_DEG, color='orange', linestyle='--', 
                           alpha=0.7, label=f'Предел ГВ ±{self.MAX_PITCH_DEG}°')
            axes[1].axhline(y=-self.MAX_PITCH_DEG, color='orange', linestyle='--', alpha=0.7)
        axes[1].set_ylabel(f'Тангаж (υ), {unit}', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper right')
        axes[1].set_title('Тангаж (ограничен ±90° из-за arcsin)', fontsize=10, style='italic')
        
        # График крена
        axes[2].plot(self.timestamps, gamma_plot, 'g-', linewidth=1.5, label='Крен (γ)')
        axes[2].set_xlabel('Время, с', fontsize=12)
        axes[2].set_ylabel(f'Крен (γ), {unit}', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='upper right')
        if unwrap:
            axes[2].set_title('Крен (с устранением разрывов)', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.show()
    
    def plot_orientation_matrix(self) -> None:
        """
        Отображает элементы матрицы ориентации во времени.
        """
        if self.C_history is None:
            raise RuntimeError("Сначала необходимо вызвать метод compute()")
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Элементы матрицы ориентации C', fontsize=16, fontweight='bold')
        
        for i in range(3):
            for j in range(3):
                ax = axes[i, j]
                ax.plot(self.timestamps, self.C_history[:, i, j], linewidth=1.5)
                ax.set_title(f'C[{i},{j}]', fontsize=11)
                ax.grid(True, alpha=0.3)
                
                if i == 2:
                    ax.set_xlabel('Время, с', fontsize=10)
        
        plt.tight_layout()
        plt.show()


def main():
    """Демонстрация работы класса Gyrovertical"""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ АЛГОРИТМА ГИРОВЕРТИКАЛИ")
    print("=" * 70)
    
    # Загрузка данных
    print("\n1. Загрузка данных датчиков...")
    acc_x = Accelerometer('data/Accelerometer_3.csv', 'x')
    acc_y = Accelerometer('data/Accelerometer_3.csv', 'y')
    acc_z = Accelerometer('data/Accelerometer_3.csv', 'z')
    
    gyro_x = Gyroscope('data/Gyroscope_3.csv', 'x')
    gyro_y = Gyroscope('data/Gyroscope_3.csv', 'y')
    gyro_z = Gyroscope('data/Gyroscope_3.csv', 'z')
    
    # Создание объекта гировертикали
    print("\n2. Инициализация гировертикали...")
    gv = Gyrovertical(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
    
    # Расчёт параметров ориентации
    print("\n3. Расчёт параметров ориентации...")
    gv.compute()
    
    # Визуализация результатов
    print("\n4. Визуализация результатов...")
    gv.plot_angles(in_degrees=True)
    
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 70)


if __name__ == '__main__':
    main()
