import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_pickle("../../data/interim/03_data_features.pkl")
# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------
df_train = df.drop(["category", "participant", "set"], axis = 1)

x = df_train.drop("label", axis = 1)
y = df_train["label"]     # for the train test split method

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42,
                                                    stratify = y)
#0.25 of data is for testing, 75% for training
# stratify is for equal distribution of all labels, so not skewed



# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------

# split all the different feature columns so we can later see if they actually help
#for the grid search later!

basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
square_features = ["acc_r", "gyr_r"]
pca_features = ["pca_1", "pca_2", "pca_3"]
#use a loop to find the columns with a specific string
time_features = [f for f in df_train.columns if "_temp_" in f]
frequency_features = [f for f in df_train.columns if ("_freq" in f) or("_pse" in f)]
cluster_features = ["cluster"]

feature_set_1 = basic_features
feature_set_2 = list(set(basic_features + square_features + pca_features)) #AVOID DUPLICATES
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + frequency_features + cluster_features))
# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------
learner = ClassificationAlgorithms
max_features = 10
selected_features, ordered_features, ordered_scores = learner.forward_selection(
    learner, max_features, x_train, y_train

)

selected_features = ['pca_1',
 'duration',
 'acc_z_freq_0.0_Hz_ws_14',
 'acc_y_temp_mean_ws_5',
 'gyr_x_freq_1.071_Hz_ws_14',
 'gyr_r_max_freq',
 'gyr_y_freq_1.429_Hz_ws_14',
 'gyr_x_temp_mean_ws_5',
 'acc_x_freq_2.143_Hz_ws_14',
 'acc_y_freq_1.786_Hz_ws_14']



# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------
possible_feature_sets = [
    feature_set_1,
    feature_set_2,
    feature_set_3,
    feature_set_4,
    selected_features
]

feature_names = [
    "Feature Set 1",
    "Feature Set 2",
    "Feature Set 3",
    "Feature Set 4",
    "Selected Features",
]

iterations = 1
score_df = pd.DataFrame()

#COPY PASTED GRID SEARCH CODE

for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = x_train[possible_feature_sets[i]]
    selected_test_X = x_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            learner, 
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            learner, selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        learner, selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        learner, selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(learner, selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])


# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------
score_df.sort_values(by ="accuracy", ascending=False)

plt.figure(figsize=(10, 10))
sns.barplot(x="model", y="accuracy", hue="feature_set", data = score_df)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1)
plt.legend(loc = "lower right")
plt.show()

# We see that feature set 4 is the best!


# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------
(
class_train_y,
class_test_y,
class_train_prob_y,
class_test_prob_y,
) = learner.random_forest(
learner, x_train[feature_set_4], y_train, x_test[feature_set_4], gridsearch=True
)

accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns

#CONFUSION MATRIX
cm = confusion_matrix(y_test, class_test_y, labels = classes)

# create confusion matrix for cm (copy pasted)
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()


# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------

participant_df = df.drop(["set", "category"], axis = 1)

## training data, all participants but A, x includes all relevant data, y includes classification
x_train = participant_df[participant_df["participant"] != "A"].drop(["label"], axis = 1)
y_train = participant_df[participant_df["participant"] != "A"]["label"]

# these are the test dfs to test our model! althought idk why x_test is necessary? 
x_test = participant_df[participant_df["participant"] == "A"].drop(["label"], axis = 1)
y_test = participant_df[participant_df["participant"] == "A"].drop(["label"], axis = 1)

x_train = x_train.drop(["participant"], axis = 1)
x_test = x_test.drop(["participant"], axis = 1)
# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------

## THE SAME CODE, JUST NEW TRAINING AND TESTING!
# (
# class_train_y,
# class_test_y,
# class_train_prob_y,
# class_test_prob_y,
# ) = learner.random_forest(
# learner, x_train[feature_set_4], y_train, x_test[feature_set_4], gridsearch=True
# )

# accuracy = accuracy_score(y_test, class_test_y)

# classes = class_test_prob_y.columns

# #CONFUSION MATRIX
# cm = confusion_matrix(y_test, class_test_y, labels = classes)

# # create confusion matrix for cm (copy pasted)
# plt.figure(figsize=(10, 10))
# plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
# plt.title("Confusion matrix")
# plt.colorbar()
# tick_marks = np.arange(len(classes))
# plt.xticks(tick_marks, classes, rotation=45)
# plt.yticks(tick_marks, classes)

# thresh = cm.max() / 2.0
# for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#     plt.text(
#         j,
#         i,
#         format(cm[i, j]),
#         horizontalalignment="center",
#         color="white" if cm[i, j] > thresh else "black",
#     )
# plt.ylabel("True label")
# plt.xlabel("Predicted label")
# plt.grid(False)
# plt.show()

# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------
