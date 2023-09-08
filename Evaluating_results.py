import matplotlib.pyplot as plt
import numpy as np
import json

# Cargar los datos desde el archivo JSON
with open('C:/Users/gaboh/UTFPR/GitHub/Toledo-Regression-Housing-predictor-IC/utils/resultados_modelos.json', 'r') as f:
    df_results = json.load(f)

models_names = df_results["models_names"]
mae_list = df_results["mae_list"]
rmse_list = df_results["rmse_list"]
pearson_list = df_results["pearson_list"]
r2_list = df_results["r2_list"]

# Configuración de la figura y los ejes con fondo gris oscuro
fig, ax = plt.subplots(figsize=(10, 8))
fig.set_facecolor('white')
ax.set_facecolor('white')

bar_width = 0.23
index = np.arange(len(models_names))

# Gráficos de barras apiladas
bar_mae = ax.bar(index, mae_list, bar_width, label='MAE')
bar_rmse = ax.bar(index + bar_width, rmse_list, bar_width, label='RMSE')
bar_pearson = ax.bar(index + 2 * bar_width, pearson_list, bar_width, label='Pearson')
bar_r2 = ax.bar(index + 3 * bar_width, r2_list, bar_width, label='R^2')

# Etiquetas, título y leyenda
ax.set_xlabel('Modelos', color='b')
ax.set_ylabel('Resultados', color='blue')
ax.set_title('Comparação Modelos', color='blue')
ax.set_xticks(index + 1.5 * bar_width)
ax.set_xticklabels(models_names, color='black')
ax.legend()

# Ajustar límites del eje y
ax.set_ylim(-1.5, 1.5)

# Mostrar valores en el eje y con tamaño de fuente reducido
def show_values_on_y_axis(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                '{:.2f}'.format(height),
                ha='center', va='bottom',
                color='black', fontsize=8)

show_values_on_y_axis(bar_mae)
show_values_on_y_axis(bar_rmse)
show_values_on_y_axis(bar_pearson)
show_values_on_y_axis(bar_r2)

# Cambiar el color de los ticks del eje y a blanco
ax.tick_params(axis='y', colors='black')

plt.tight_layout()
plt.show()

