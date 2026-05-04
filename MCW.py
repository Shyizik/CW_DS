import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from sklearn.metrics import mean_squared_error, r2_score

# 1. Підготовка даних
np.random.seed(42)
x = np.linspace(0, 10, 25)
y_true = 2.2 * x + 4.5  # Еталонна лінія

# 2. Налаштування візуалізації
fig, ax = plt.subplots(figsize=(11, 7))
plt.subplots_adjust(bottom=0.25)  # Місце для повзунка та кнопки


# 3. Математичне ядро (МНК)
def fit_lsm(x_d, y_d):
    # Формування матриці плану (стовпець X та стовпець одиниць)
    A = np.vstack([x_d, np.ones(len(x_d))]).T
    # Розв'язання системи рівнянь матричним методом
    return np.linalg.lstsq(A, y_d, rcond=None)[0]


# 4. Початковий стан
init_noise = 2.0
y_noisy = y_true + np.random.normal(0, init_noise, size=len(x))
a, b = fit_lsm(x, y_noisy)
y_pred = a * x + b

# 5. Створення графічних об'єктів
points, = ax.plot(x, y_noisy, 'ro', alpha=0.6, label='Дані вимірів')
line, = ax.plot(x, y_pred, 'b-', lw=2.5, label='Лінія МНК')
# Візуалізація відхилень (залишків)
residuals = ax.vlines(x, y_pred, y_noisy, colors='gray', linestyles='--', alpha=0.4, label='Залишки')

# Текстовий блок для метрик (R², RMSE)
stats_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))


def update_stats(y_n, y_p):
    r2 = r2_score(y_n, y_p)
    rmse = np.sqrt(mean_squared_error(y_n, y_p))
    stats_text.set_text(f'R²: {r2:.3f}\nRMSE: {rmse:.2f}')


update_stats(y_noisy, y_pred)

# Оформлення осей
ax.set_ylim(-5, 35)
ax.set_title('Аналіз стійкості МНК до шумів', fontsize=14)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# 6. Інтерактивні елементи (UI)
ax_noise = plt.axes([0.2, 0.12, 0.6, 0.03])
s_noise = Slider(ax_noise, 'Рівень шуму ', 0.0, 15.0, valinit=init_noise, color='#2ecc71')

ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
btn_reset = Button(ax_reset, 'Reset', color='#ecf0f1', hovercolor='tomato')


# 7. Обробка подій
def update(val):
    noise_val = s_noise.val
    np.random.seed(42)  # Для фіксації точок при певному рівні шуму
    new_y = y_true + np.random.normal(0, noise_val, size=len(x))

    # Перерахунок моделі
    new_a, new_b = fit_lsm(x, new_y)
    new_pred = new_a * x + new_b

    # Оновлення графіки
    points.set_ydata(new_y)
    line.set_ydata(new_pred)

    # Перерахунок ліній залишків
    new_segments = [np.array([[xi, yi], [xi, ypi]]) for xi, yi, ypi in zip(x, new_y, new_pred)]
    residuals.set_segments(new_segments)

    update_stats(new_y, new_pred)
    fig.canvas.draw_idle()


def reset(event):
    s_noise.reset()


s_noise.on_changed(update)
btn_reset.on_clicked(reset)

plt.show()