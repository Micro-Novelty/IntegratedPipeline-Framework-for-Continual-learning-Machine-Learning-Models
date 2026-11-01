from pipeline_setup import one_run
from pipeline_setup import train_run
import pandas

# EXAMPLE REGRESSION TASK:
df = pd.read_csv("PJME_hourly.csv", parse_dates=["Datetime"], index_col="Datetime")
df = df.sort_index()
daily = df.resample("D").sum()

# calender features (day of week, month, is weekend) (suit to what your dataset actually has)
daily["dayofweek"] = daily.index.dayofweek     
daily["month"] = daily.index.month
daily["is_weekend"] = (daily.index.dayofweek >= 5).astype(int)

# lag features (in days, not hours) (suit to what your dataset actually has)
daily["lag_1"] = daily["PJME_MW"].shift(1)      # yesterday (lag_1 will be used in feature_cols)
daily["lag_7"] = daily["PJME_MW"].shift(7)      # last week (same day) (lag_7 will be used in feature_cols)
daily["lag_14"] = daily["PJME_MW"].shift(14)    # 2 weeks ago (lag_14 will be used in feature_cols)

# Rolling statistics (suit to what your dataset actually has)
daily["rolling_mean_7"] = daily["PJME_MW"].rolling(7).mean() # (rolling_mean_7 will be used in feature_cols)
daily["rolling_std_7"] = daily["PJME_MW"].rolling(7).std()   # (rolling_std_7 will be used in feature_cols)
# delete the first row that contains NaN values due to lag and rolling calculations
daily = daily.dropna()

# Feature columns: a set of list thatcontains the features you want to use for training and prediction.
feature_cols = ["PJME_MW", "dayofweek", "month", "is_weekend",
            "lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_std_7"]

X, y = '<load your X and y samples> # TASK A (e.g task A has shape (500, 8))
# config is needed for prediction on regression tasks:
s_X, s_y, config = get_dataset(daily, feature_cols, target_col_name="PJME_MW",
                                  window_size=30, horizon=8) # TASK B is regression type (e.g task B has shape (500, 8))
t_X, t_y = '<load your third X and y samples> TASK C (e.g task C must have the same shape as task A and B (500, 8) for prediction later)

db_path = 'model_memory_storage.db' # path to your sqlite database, customizable name (must put .db in the end).

# ========= TESTING BLOCK (for validation test) =============
# FOR TESTING ON CLASSIFICATION TASK 
results, chosen_label, confidence = one_run(db_path, X, y, task='classification', pred_config=None, 
                                        label_csv='ManualsTraining.txt', feature_cols=None)

# FOR TESTING ON REGRESSION TASK
results, chosen_label, confidence = one_run(db_path, X, y, task='regression', pred_config=config, 
                            label_csv='energy_training.txt', feature_cols=feature_cols)

# ========== CONTINUOUS TRAIN AND PREDICTION OVER REAL SAMPLES =============
# TRAIN ON TASK A
# train_run() function will automatically load and save memory including labels and samples.
train_run(db_path, X, y, task='classification', pred_config=None, 
          label_csv='ManualsTraining.txt', feature_cols=None)

# TRAIN ON TASK B 
train_run(db_path, s_X, s_y, task='regression', pred_config=config, 
           label_csv='energy_training.txt', feature_cols=feature_cols)

# PREDICT ON TASK C (TASK C must have correlation with task A or B)
# with_training=False will not trigger any Training.
# eventually since you have already Trained the model on task A and B, its memory name, X and Y samples, label_map, feature_cols and config is saved to database, so prediction is now automatic, allowing users to continuously predict over the previously trained samples.
results, chosen_label, confidence = one_run(db_path, t_X, t_y, with_training=False) 
