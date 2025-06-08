# Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from itertools import cycle
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
import mlflow

# Funciones auxiliares
from tca_kedrito.utils.aux_funcs import get_numeric_and_categorical_columns, date_extraction


def preprocess_reservation_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[['h_res_fec', 'h_fec_lld', 'h_num_adu', 'h_num_men', 'h_num_noc', 'ID_Tipo_Habitacion', 'ID_Paquete', 'h_can_res', 'cluster_gmm']].copy()
    df['h_res_fec'] = pd.to_datetime(df['h_res_fec'])
    df['h_fec_lld'] = pd.to_datetime(df['h_fec_lld'])
    df['ID_Tipo_Habitacion'] = df['ID_Tipo_Habitacion'].replace('Otro', '0')
    df['h_can_res'] = df['h_can_res'].replace('DI', '05')
    date_extraction(df, 'h_res_fec', 'dia', 'reservacion')
    date_extraction(df, 'h_fec_lld', 'dia', 'entrada')
    date_extraction(df, 'h_res_fec', 'mes', 'reservacion')
    date_extraction(df, 'h_fec_lld', 'mes', 'entrada')
    df.drop(columns=['h_res_fec', 'h_fec_lld'], inplace=True)

    # Variables cíclicas: días de la semana
    day_map = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    df['dia_reservacion'] = df['dia_reservacion'].map(day_map).astype(int)
    df['dia_entrada'] = df['dia_entrada'].map(day_map).astype(int)

    # Variables cíclicas: meses del año
    mes_map = {
        'January': 0, 'February': 1, 'March': 2, 'April': 3,
        'May': 4, 'June': 5, 'July': 6, 'August': 7,
        'September': 8, 'October': 9, 'November': 10, 'December': 11
    }
    df['mes_reservacion'] = df['mes_reservacion'].map(mes_map)
    df['mes_entrada'] = df['mes_entrada'].map(mes_map)

    # Identificar y formatear columnas numéricas y categóricas
    numerical_cols, categorical_cols = get_numeric_and_categorical_columns(df, numeric_as_category=['ID_Paquete', 'cluster_gmm', 'mes_entrada', 'mes_reservacion', 'dia_entrada', 'dia_reservacion'])

    df = df[numerical_cols + categorical_cols]

    return df

def reduce_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    # Obtener los 2 valores más frecuentes para 'ID_Tipo_Habitacion'
    top2_habitacion = df['ID_Tipo_Habitacion'].value_counts().nlargest(2).index.tolist()
    df['ID_Tipo_Habitacion'] = df['ID_Tipo_Habitacion'].apply(lambda x: x if x in top2_habitacion else 'Otro_Habitacion')

    # Convertir 'h_can_res' a numérico para encontrar los 2 valores más frecuentes
    df['h_can_res_numeric'] = pd.to_numeric(df['h_can_res'], errors='coerce')
    top2_res = df['h_can_res_numeric'].dropna().value_counts().nlargest(2).index.tolist()
    top2_res_str = [str(int(x)) for x in top2_res]
    df['h_can_res'] = df['h_can_res'].apply(lambda x: x if str(x) in top2_res_str else 'Otro_Res')
    df.drop(columns=['h_can_res_numeric'], inplace=True)

    numerical_cols, categorical_cols = get_numeric_and_categorical_columns(df, numeric_as_category=['ID_Paquete', 'cluster_gmm', 'mes_entrada', 'mes_reservacion', 'dia_entrada', 'dia_reservacion', 'ID_Tipo_Habitacion', 'h_can_res'])
    numerical_cols, categorical_cols
    df = df[numerical_cols + categorical_cols]

    return df

def train_svm_model(df: pd.DataFrame):
    # Separar variables predictoras y etiquetas
    X = df.drop(columns=['cluster_gmm'])
    y = df['cluster_gmm']

    # Preprocesamiento
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['h_num_adu', 'h_num_men', 'h_num_noc']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['ID_Paquete', 'dia_entrada', 'dia_reservacion', 'mes_entrada', 'mes_reservacion', 'ID_Tipo_Habitacion', 'h_can_res'])
    ])

    # Modelo SVM
    model = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape='ovo'))
    ])

    # División del conjunto
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="SVM Classification"):
        # Entrenamiento
        model.fit(X_train, y_train)

        # Evaluación
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        print(f"Accuracy: {acc:.3f}")
        print(f"F1 Score (macro): {f1:.3f}")

        # Mlflow Log de parámetros y métricas
        mlflow.log_param("kernel", "rbf")
        mlflow.log_param("C", 1.0)
        mlflow.log_param("gamma", "scale")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        # Mlflow Log del modelo
        mlflow.sklearn.log_model(model, "svm_model")

        return model
