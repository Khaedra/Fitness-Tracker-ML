import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis #EXTERNAL CLASSES FROM GITHUB
from TemporalAbstraction import NumericalAbstraction


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenet.pkl")

predictor_columns = list(df.columns[:6])

plt.style.use("fivethirtyeight") 
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2



# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in predictor_columns: 
    df[col] = df[col].interpolate()
    

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0] #20 seconds!
duration.seconds

#find average duration
for s in df["set"].unique():
    end =  df[df["set"] == s].index[-1] 
    start = df[df["set"] == s].index[0]
    duration = end - start
    
    df.loc[(df["set"] == s), "duration"] = duration.seconds #create new duration column
    
duration_df = df.groupby(["category"])["duration"].mean()
duration_df.iloc[0] / 5
duration_df.iloc[1] / 10 #average time per rep for heavy and medium sets


# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
df_lowpass = df.copy()

LowPass = LowPassFilter() #constructer for external class
fs = 1000 / 200 #5 measurements per second
cutoff = 1.3
df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order = 5)

for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order = 5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col+ "_lowpass"] #override original columns with new values

df_lowpass

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

#determine optimal amount of principal compnents, using given function
pca_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)
#found that 3 is the optimal number, using elbow method

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)
#essentially summarized the 6 numerical columns into 3

subset = df_pca[df_pca["set"] == 35]
subset[["pca_1", "pca_2","pca_3"]].plot()



# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] **2 + df_squared["acc_y"] **2 + df_squared["acc_z"]**2
gyr_r = df_squared["gyr_x"] **2 + df_squared["gyr_y"] **2 + df_squared["gyr_z"]**2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)    #calculate scalar and make new column

subset = df_squared[df_squared["set"] == 14]
subset[["acc_r", "gyr_r"]].plot(subplots = True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
