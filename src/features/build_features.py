import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis #EXTERNAL CLASSES FROM GITHUB
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

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
df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyr_r"] #add other numerical columns
ws = int(1000/200) #window size = 1 second

for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")
    
# different exercises may use data from previous ones since it just looks back 5 values
#split into different sets

df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset) #add subset data into list?
    
df_temporal = pd.concat(df_temporal_list) #concat it all to one df

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
        

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index()

FreqAbs = FourierTransformation()
 
fs = int(1000/200)       #sampling rate, number of samples per second
ws = int(2800/200)  #window size, average length of rep in ms/200 = 14

#df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)
df_freq_list = []
for s in df_freq["set"].unique():
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop = True)


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq = df_freq.dropna()
#get rid of 50% of data, prevents overfitting in the long run?
df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []

for k in k_values: #loop over appropriate k values
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters = k, n_init = 20, random_state = 0)
    cluster_labels = kmeans.fit_predict(subset) #fit predict assigns data in each row to a cluster
    inertias.append(kmeans.inertia_) #in documentation, sum of squared distance to cluster center

plt.figure(figsize = (10, 10))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.show()
#clear elbow at 5

kmeans = KMeans(n_clusters = 5, n_init = 20, random_state = 0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

#plotting the cluster on a 3d graph, we can see clear groups
# plotting the different excercises on the 3d graph, they correspond to cluster groups
#however, some exercises are similar so 2 in one cluster.

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
df_cluster.to_pickle("../../data/interim/03_data_features.pkl")