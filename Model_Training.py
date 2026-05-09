import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

MODEL_PATH = "model.pkl"

def preprocess(df):
    df = df.copy()
    df = df.dropna()

    categorical_cols = [
        'person_home_ownership',
        'loan_intent',
        'loan_grade',
        'cb_person_default_on_file'
    ]

    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    return df, le_dict


def train_and_save_model(df):
    df, le_dict = preprocess(df)

    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    lr_pred = lr.predict_proba(X_test)[:, 1]
    rf_pred = rf.predict_proba(X_test)[:, 1]

    lr_auc = roc_auc_score(y_test, lr_pred)
    rf_auc = roc_auc_score(y_test, rf_pred)

    if rf_auc > lr_auc:
        best_model = rf
        model_name = "Random Forest"
        best_auc = rf_auc
    else:
        best_model = lr
        model_name = "Logistic Regression"
        best_auc = lr_auc

    with open(MODEL_PATH, "wb") as f:
        pickle.dump((best_model, le_dict, X.columns), f)

    return model_name, best_auc


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(df, model, le_dict, columns):
    df = df.copy()
    df = df.dropna()

    for col, le in le_dict.items():
        df[col] = le.transform(df[col])

    X = df[columns]

    df['default_probability'] = model.predict_proba(X)[:, 1]

    def risk(p):
        if p < 0.3:
            return "Low"
        elif p < 0.7:
            return "Medium"
        else:
            return "High"

    df['risk_category'] = df['default_probability'].apply(risk)

    return df