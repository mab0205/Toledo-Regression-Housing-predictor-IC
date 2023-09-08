from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from scipy.stats import pearsonr

def resultsModels(y_true, predictions):
    mae = mean_absolute_error(y_true, predictions)
    #print("MAE:", mae)

    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    #print("RMSE:", rmse)

    pearson_corr, _ = pearsonr(y_true, predictions)
    #print("Pearson Correlation:", pearson_corr)

    r2 = r2_score(y_true, predictions)
    #print("R2 Score:", r2)

    return mae, rmse, pearson_corr, r2
    
