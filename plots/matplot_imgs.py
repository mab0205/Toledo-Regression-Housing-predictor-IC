import matplotlib.pyplot as plt
import seaborn as sns

def boxplot(data, columnas):
    num_columnas = len(columnas)
    num_filas = (num_columnas + 2) // 3  # Calcula el número de filas necesarias

    plt.figure(figsize=(15, 5 * num_filas))

    for i, column in enumerate(columnas, 1):
        plt.subplot(num_filas, 3, i)  # Divide en filas y columnas
        sns.boxplot(data[column])
        plt.title(column)
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

def coMatrix(numeric_data):
    # Calcular la matriz de correlación
    correlation_matrix = numeric_data.corr()

    # Visualizar la matriz de correlación
    plt.figure(figsize = (12, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title('Matriz de Correlación')
    plt.show()

def plotFeatures(df, columns):
    color1 = 'blue'  # o cualquier otro color válido
    df[columns].hist(bins=100, figsize=(30, 12), color=color1)
    plt.show()