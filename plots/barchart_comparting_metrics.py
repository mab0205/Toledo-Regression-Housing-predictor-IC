import matplotlib.pyplot as plt

def barchart(models, mae_list, rmse_list, pearson_list, r2_list):
    # Plot MAE
    plt.figure(figsize=(10, 6))
    plt.bar(models, mae_list, color='blue')
    plt.xlabel('Models')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error (MAE) ')
    plt.xticks(rotation=45)
    plt.show()

    # Plot RMSE
    plt.figure(figsize=(10, 6))
    plt.bar(models, rmse_list, color='green')
    plt.xlabel('Models')
    plt.ylabel('RMSE')
    plt.title('Root Mean Squared Error (RMSE) ')
    plt.xticks(rotation=45)
    plt.show()

    # Plot Pearson Correlation
    plt.figure(figsize=(10, 6))
    plt.bar(models, pearson_list, color='orange')
    plt.xlabel('Models')
    plt.ylabel('Pearson Correlation')
    plt.title('Pearson Correlation ')
    plt.xticks(rotation=45)
    plt.show()

    # Plot R2 Score
    plt.figure(figsize=(10, 6))
    plt.bar(models, r2_list, color='purple')
    plt.xlabel('Models')
    plt.ylabel('R2 Score')
    plt.title('R2 Score')
    plt.xticks(rotation=45)
    plt.show()