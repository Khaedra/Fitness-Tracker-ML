import pandas as pd 
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_file_acc = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")

single_file_gyr = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")
# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
# use glob library
files = glob("../../data/raw/MetaMotion/*.csv") #all csv files at this directory 
len(files)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
data_path = "../../data/raw/MetaMotion/"
f = files[0] #get the first file!

participant = f.split("-")[0].replace(data_path, "") #Participant name
label = f.split("-")[1] #which exercise
category = f.split("-")[2].rstrip("123") #rightstrip set number

df = pd.read_csv(f)

df["participant"] = participant #add new columns
df["label"] = label
df["category"] = category
# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

acc_df = pd.DataFrame(); 
gyr_df = pd.DataFrame(); 

acc_set = 1; 
gyr_set = 1; 

for f in files:
    
    participant = f.split("-")[0].replace(data_path, "") #Participant name
    label = f.split("-")[1] #which exercise
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019") #rightstrip set number
    
    df = pd.read_csv(f)
    
    df["participant"] = participant #add new columns
    df["label"] = label
    df["category"] = category
    
    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df]) #if accelerometer file, append to master acc file
    if "Gyroscope" in f:
        df["set"] = gyr_set
        gyr_set += 1
        gyr_df = pd.concat([gyr_df, df]) #if accelerometer file, append to master acc file



# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------


acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit = "ms") #set index to the time!
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit = "ms")
#now remove time columns
del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]



# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------
files = glob("../../data/raw/MetaMotion/*.csv")
def read_data_from_files(files):
    
    acc_df = pd.DataFrame(); 
    gyr_df = pd.DataFrame(); 

    acc_set = 1; 
    gyr_set = 1; 

    for f in files:
        
        participant = f.split("-")[0].replace(data_path, "") #Participant name
        label = f.split("-")[1] #which exercise
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019") #rightstrip set number
        
        df = pd.read_csv(f)
        
        df["participant"] = participant #add new columns
        df["label"] = label
        df["category"] = category
        
        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df]) #if accelerometer file, append to master acc file
        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df]) #if accelerometer file, append to master acc file

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit = "ms") #set index to the time!
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit = "ms")
    #now remove time columns
    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]
    
    return acc_df, gyr_df

acc_df, gyr_df = read_data_from_files(files)

#YOU CAN DELETE EVERYTHING EXCEPT THIS FUNCTION TO MAKE IT SHORTER
# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
data_merged = pd.concat([acc_df.iloc[:,:3], gyr_df], axis=1) #columnwise concat, row = 0
# take the first 3 columns of acc so there is no redundancy
#missing data, gyro takes more frequent measurements

data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz
#gyroscope twice as fast, make them equal

sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "label" : "last",
    "category" : "last",
    "participant" :"last",
    "set" : "last"
}
#dictionary to specify which function to apply to each row

data_merged[:100].resample(rule = "200ms").apply(sampling)
#group data by every 200 ms, apply the dictioanry function to each row.
#this eliminates missing data since you are taking average of a time period. 


#now to save computing power, split into days, apply resample to each day, bring back together
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
data_resampled = pd.concat([df.resample(rule = "200ms").apply(sampling).dropna() for df in days])
#loop over days, for each df in days, apply resample rule, drop nulls.

data_resampled["set"] = data_resampled["set"].astype("int")
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
#export to data->interim as a pickle file

data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")
#export df in serialized format, reading will be the exact same as it was exported
#good for time stamps

#gonna open it up in a python file again to edit, pickle better than csv 