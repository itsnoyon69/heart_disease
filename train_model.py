import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv("heart.csv")

# =====================================
# FEATURES AND TARGET
# =====================================

X = df.drop("target", axis=1)
y = df["target"]

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# SCALING
# =====================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

# =====================================
# MODEL
# =====================================

model = LogisticRegression()

# =====================================
# TRAIN MODEL
# =====================================

model.fit(X_train, y_train)

# =====================================
# SAVE MODEL
# =====================================

pickle.dump(
    model,
    open("model.pkl", "wb")
)

# =====================================
# SAVE SCALER
# =====================================

pickle.dump(
    scaler,
    open("scaler.pkl", "wb")
)

print("Model Saved Successfully")