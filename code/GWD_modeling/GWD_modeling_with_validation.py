import os
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
os.chdir(sys.path[0])

RMSE_color = "#a2c3e1"
RMSE_edge_color = "#0b5298"
MAE_color = "#fbcfa0"
MAE_edge_color = "#f09b4b"

with open('./config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

predict_tar = config.get('predict_tar')
predict_year = config.get('predict_year')
features = config.get('features')
lag_factor = config.get('lag_factor')
acc_factor = config.get('acc_factor')
total_factor = config.get('total_factor')


def prepare_data():
    df = pd.read_csv('./output/station_groundwater_wd_wheatmaize.csv')
    df = df[df['层次'] == '浅层']
    df['Year'] = df['YearMonth'].str[:4].astype(int)
    df['Month'] = df['YearMonth'].str[5:7].astype(int)
    df = df.sort_values(['测站编', 'YearMonth']).reset_index(drop=True)
    df[lag_factor] = np.nan
    
    for s in df['测站编'].unique():
        idx = df[df['测站编'] == s].index.tolist()
        for i in range(len(idx)):
            if i >= 1:
                df.loc[idx[i], 'WB_lag1'] = df.loc[idx[i-1], 'WaterBarr']
            if i >= 2:
                df.loc[idx[i], 'WB_lag2'] = df.loc[idx[i-2], 'WaterBarr']
    
    for v in acc_factor:
        for k in [2, 3]:
            df[f'{v}_cum{k}'] = np.nan
    
    for s in df['测站编'].unique():
        idx = df[df['测站编'] == s].index.tolist()
        for i, idxi in enumerate(idx):
            w2 = idx[max(0, i-1):i+1]
            for v in acc_factor:
                df.loc[idxi, f'{v}_cum2'] = df.loc[w2, v].sum() / 2
            w3 = idx[max(0, i-2):i+1]
            for v in acc_factor:
                df.loc[idxi, f'{v}_cum3'] = df.loc[w3, v].sum() / 3
    
    df = df[df['Year'] >= predict_year].reset_index(drop=True)
    df = df.dropna(subset=total_factor).reset_index(drop=True)
    return df


def evaluate(y_true, y_pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }


def train_models(X_train, y_train, X_val, y_val):
    models = {
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'SVM': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'DT': DecisionTreeRegressor(random_state=42, max_depth=10),
        'RF': RandomForestRegressor(n_estimators=120, random_state=42, max_depth=12)
    }
    results = {}
    predictions = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        results[name] = evaluate(y_val, y_pred)
        predictions[name] = y_pred
    
    return results, predictions


def leave_one_year_all(df):
    years = sorted(df['Year'].unique())
    all_results = {}
    rf_preds = {}
    y_trues = {}
    
    for test_year in years:
        train_df = df[df['Year'] != test_year]
        val_df = df[df['Year'] == test_year]
        X_train = train_df[features].values
        y_train = train_df[predict_tar].values
        X_val = val_df[features].values
        y_val = val_df[predict_tar].values
        
        results, predictions = train_models(X_train, y_train, X_val, y_val)
        all_results[test_year] = results
        rf_preds[test_year] = predictions['RF']
        y_trues[test_year] = y_val
    
    return all_results, rf_preds, y_trues


def leave_location_kfold(df, n_splits=5):
    np.random.seed(42)
    stations = df['测站编'].unique()
    np.random.shuffle(stations)
    
    # 将井分成5折
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    fold_rf_preds = []
    fold_y_trues = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(stations)):
        train_stations = stations[train_idx]
        test_stations = stations[test_idx]
        
        train_df = df[df['测站编'].isin(train_stations)]
        val_df = df[df['测站编'].isin(test_stations)]
        
        X_train = train_df[features].values
        y_train = train_df[predict_tar].values
        X_val = val_df[features].values
        y_val = val_df[predict_tar].values
        
        results, predictions = train_models(X_train, y_train, X_val, y_val)
        fold_results.append(results)
        fold_rf_preds.append(predictions['RF'])
        fold_y_trues.append(y_val)
    
    return fold_results, fold_rf_preds, fold_y_trues


def plot_figure(df, results_year, results_loc, rf_preds_year, y_trues_year, rf_preds_loc, y_trues_loc):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Arial'],
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False
    })
    
    fig = plt.figure(figsize=(6.9, 6.9))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # (a) 
    ax_a = fig.add_subplot(gs[0, 0])
    all_preds = np.concatenate(list(rf_preds_year.values()))
    all_true = np.concatenate(list(y_trues_year.values()))
    ax_a.scatter(all_preds, all_true, alpha=0.4, s=10, c='black', edgecolors='black', linewidth=0.3)
    min_val = min(all_true.min(), all_preds.min())
    max_val = max(all_true.max(), all_preds.max())
    ax_a.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line', alpha=0.7)
    ax_a.legend(frameon=False, fontsize=9, loc='upper left')
    ax_a.set_xlabel('Predicted GWD (m)')
    ax_a.set_ylabel('Observed GWD (m)')
    ax_a.text(0.02, 1.02, 'a', transform=ax_a.transAxes, fontweight='bold', fontsize=12)
    rmse_mean = np.mean([np.sqrt(mean_squared_error(y_trues_year[y], rf_preds_year[y])) for y in rf_preds_year.keys()])
    mae_mean = np.mean([mean_absolute_error(y_trues_year[y], rf_preds_year[y]) for y in rf_preds_year.keys()])
    ax_a.text(0.02, 0.85, f'Mean RMSE = {rmse_mean:.3f} m', transform=ax_a.transAxes, fontsize=9)
    ax_a.text(0.02, 0.78, f'Mean MAE = {mae_mean:.3f} m', transform=ax_a.transAxes, fontsize=9)
    
    # (b) 
    ax_b = fig.add_subplot(gs[0, 1])
    models = list(results_year[list(results_year.keys())[0]].keys())
    years = sorted(results_year.keys())
    x = np.arange(len(models))
    width = 0.25
 
    rmse_matrix = np.array([[results_year[y][m]['RMSE'] for m in models] for y in years])
    mae_matrix = np.array([[results_year[y][m]['MAE'] for m in models] for y in years])

    rmse_mean = np.mean(rmse_matrix, axis=0)
    rmse_std = np.std(rmse_matrix, axis=0)
    mae_mean = np.mean(mae_matrix, axis=0)
    mae_std = np.std(mae_matrix, axis=0)
    
    bars1 = ax_b.bar(x - width/2, rmse_mean, width, label='RMSE', color=RMSE_color, edgecolor=RMSE_edge_color, linewidth=0.8, yerr=rmse_std, capsize=3, error_kw={'ecolor': RMSE_edge_color, 'linewidth': 1})
    ax_b.set_ylabel('RMSE (m)', color=RMSE_edge_color)
    ax_b.set_ylim(0, 10)
    ax_b.set_yticks([0, 1, 2, 3, 4,5, 6,7,8,9,10])
    ax_b.tick_params(axis='y', labelcolor=RMSE_edge_color)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(models, ha='center')
    ax_b.text(0.02, 1.02, 'b', transform=ax_b.transAxes, fontweight='bold', fontsize=12)
    
    
    # 双轴MAE
    ax_b2 = ax_b.twinx()
    bars2 = ax_b2.bar(x + width/2, mae_mean, width, label='MAE', color=MAE_color, edgecolor=MAE_edge_color, linewidth=0.8, yerr=mae_std, capsize=3, error_kw={'ecolor': MAE_edge_color, 'linewidth': 1})
    ax_b2.set_ylabel('MAE (m)', color=MAE_edge_color)
    ax_b2.tick_params(axis='y', labelcolor=MAE_edge_color)
    ax_b2.set_ylim(0, 10)
    ax_b2.set_yticks([0, 1, 2, 3, 4,5, 6,7,8,9,10])

    ax_b.legend([bars1, bars2], ['RMSE', 'MAE'], frameon=False, fontsize=8, loc='upper right')
    
    # (c) 
    ax_c = fig.add_subplot(gs[1, 0])
    all_preds_loc = np.concatenate(rf_preds_loc)
    all_true_loc = np.concatenate(y_trues_loc)
    ax_c.scatter(all_preds_loc, all_true_loc, alpha=0.4, s=10, c='black', edgecolors='black', linewidth=0.3)
    min_val = min(all_true_loc.min(), all_preds_loc.min())
    max_val = max(all_true_loc.max(), all_preds_loc.max())
    ax_c.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line', alpha=0.7)
    ax_c.legend(frameon=False, fontsize=9, loc='upper left')
    ax_c.set_xlabel('Predicted GWD (m)')
    ax_c.set_ylabel('Observed GWD (m)')
    ax_c.text(0.02, 1.02, 'c', transform=ax_c.transAxes, fontweight='bold', fontsize=12)
    rmse_mean_loc = np.mean([np.sqrt(mean_squared_error(y_trues_loc[i], rf_preds_loc[i])) for i in range(len(rf_preds_loc))])
    mae_mean_loc = np.mean([mean_absolute_error(y_trues_loc[i], rf_preds_loc[i]) for i in range(len(rf_preds_loc))])
    ax_c.text(0.02, 0.85, f'Mean RMSE = {rmse_mean_loc:.3f} m', transform=ax_c.transAxes, fontsize=9)
    ax_c.text(0.02, 0.78, f'Mean MAE = {mae_mean_loc:.3f} m', transform=ax_c.transAxes, fontsize=9)
    
    # (d) 
    ax_d = fig.add_subplot(gs[1, 1])
    models = list(results_loc[0].keys())
    x = np.arange(len(models))
    rmse_matrix_loc = np.array([[r[m]['RMSE'] for m in models] for r in results_loc])
    mae_matrix_loc = np.array([[r[m]['MAE'] for m in models] for r in results_loc])
    
    rmse_mean = np.mean(rmse_matrix_loc, axis=0)
    rmse_std = np.std(rmse_matrix_loc, axis=0)
    mae_mean = np.mean(mae_matrix_loc, axis=0)
    mae_std = np.std(mae_matrix_loc, axis=0)
    
    bars1 = ax_d.bar(x - width/2, rmse_mean, width, label='RMSE', color=RMSE_color, edgecolor=RMSE_edge_color, linewidth=0.8, yerr=rmse_std, capsize=3, error_kw={'ecolor': RMSE_edge_color, 'linewidth': 1})
    ax_d.set_ylabel('RMSE (m)', color=RMSE_edge_color)
    ax_d.set_ylim(0, 10)
    ax_d.set_yticks([0, 1, 2, 3, 4,5, 6,7,8,9,10])
    ax_d.tick_params(axis='y', labelcolor=RMSE_edge_color)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(models, ha='center')
    ax_d.text(0.02, 1.02, 'd', transform=ax_d.transAxes, fontweight='bold', fontsize=12)
    
    ax_d2 = ax_d.twinx()
    bars2 = ax_d2.bar(x + width/2, mae_mean, width, label='MAE', color=MAE_color, edgecolor=MAE_edge_color, linewidth=0.8, yerr=mae_std, capsize=3, error_kw={'ecolor': MAE_edge_color, 'linewidth': 1})
    ax_d2.set_ylabel('MAE (m)', color=MAE_edge_color)
    ax_d2.tick_params(axis='y', labelcolor=MAE_edge_color)
    ax_d2.set_ylim(0, 10)
    ax_d2.set_yticks([0, 1, 2, 3, 4, 5,6,7,8,9,10])
    
    ax_d.legend([bars1, bars2], ['RMSE', 'MAE'], frameon=False, fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('./Figure9.pdf', dpi=300, bbox_inches='tight')
    # plt.show()


df = prepare_data()

# leaving one year validation
results_year, rf_preds_year, y_trues_year = leave_one_year_all(df)

# 5folds leaving stations validation
results_loc, rf_preds_loc, y_trues_loc = leave_location_kfold(df, n_splits=5)

# draw plots
plot_figure(df, results_year, results_loc, rf_preds_year, y_trues_year, rf_preds_loc, y_trues_loc)

print("done")