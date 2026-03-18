import joblib
import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# paths
MODEL_PATH = "Model/xgb_salary_pipeline_best.pkl"
PREPROCESSOR_PATH = "Model/OneHotEncoder_categories.pkl"
# TEST_DATA_PATH = r"data/processed/test.csv"

def load_model():
    model = joblib.load(MODEL_PATH)
    return model

def load_preprocessor():
    with open(PREPROCESSOR_PATH, 'rb') as f:
        encoder = pickle.load(f)
    return encoder

def expbucket(x):
    if x <= 2:
        return 'junior'
    elif x <= 5 and x > 2:
        return 'mid'
    elif x <= 10 and x > 5:
        return 'senior'
    else:
        return 'lead'
    
def encode_remote(x):
    if 'remote' in x.lower():
        return 1
    else:
        return 0
    
def preprocess(df, encoder):
    exp_order = {'junior': 0, 'mid': 1, 'senior': 2, 'lead': 3}
    edu_order = {'bootcamp/self-taught': 0, 'associate': 1, 'bachelor': 2, 'master': 3, 'phd': 4}
    company_size_order = {'startup': 0, 'sme': 1, 'mid': 2, 'enterprise': 3, 'big': 4}

    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'str':
            df[col] = df[col].apply(lambda x: x.lower())

    df['bucket_of_experience'] = df['years_of_experience'].apply(expbucket)
    df['bucket_of_experience'] = df['years_of_experience'].map(exp_order)
    df['experience_level'] = df['experience_level'].apply(lambda x: x.split(' ')[0])
    df['experience_level'] = df['experience_level'].map(exp_order).fillna(0)
    df['company_size'] = df['company_size'].apply(lambda x: x.split(' ')[0])
    df['company_size'] = df['company_size'].map(company_size_order)
    df['education_required'] = df['education_required'].apply(lambda x: x.split("'")[0])
    df['education_required'] = df['education_required'].map(edu_order).fillna(0)
    df['remote_work'] = df['remote_work'].apply(encode_remote)

    onehot_encoder = encoder

    str_cols = df.select_dtypes(include=['object', 'str']).columns
    print("String columns to encode:", str_cols)
    encoded = onehot_encoder.transform(df[str_cols])
    df = pd.concat([df, pd.DataFrame(encoded.toarray(), columns=onehot_encoder.get_feature_names_out(str_cols))], axis=1)
    df.drop(columns=str_cols, inplace=True)
    
    return df

model = load_model()
onehot_encoder = load_preprocessor()

def predict(model, df):
    x = preprocess(df, onehot_encoder)
    y_pred = model.predict(x)
    return y_pred[0]


# model = load_model()
# test = pd.read_csv("test.csv")

# x,y = test.drop(columns=['annual_salary_usd']), test['annual_salary_usd']

# # x_preprocessed = preprocess(x)

# y_pred = model.predict(x)

# mse = mean_squared_error(y, y_pred)
# rmse = mse ** 0.5
# mae = mean_absolute_error(y, y_pred)
# r2 = r2_score(y, y_pred)

# print("Minimum actual salary:", min(y))
# print("Maximum actual salary:", max(y))
# print("MSE:", mse)
# print("RMSE:", rmse)
# print("MAE:", mae)
# print("R2:", r2)