import pandas as pd
import numpy as np
import re
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from tca_kedrito.utils.aux_funcs import get_numeric_and_categorical_columns, date_extraction, cramers_v, cramers_v_matrix, encontrar_codo, drop_columns_inplace, remove_outliers_iqr, convert_columns_type

def load_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(['h_correo_e', 'h_nom'], axis=1)
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.set_index(df['ID_Reserva'])
    df.drop(['ID_Reserva'], axis=1, inplace=True)

    # Convertir fechas
    columnas_fechas = [
        'Fecha_hoy', 'h_fec_reg_ok', 'h_fec_reg_okt', 'h_res_fec_ok',
        'h_res_fec_okt', 'h_fec_lld_ok', 'h_fec_lld_okt',
        'h_ult_cam_fec_ok', 'h_ult_cam_fec_okt', 'h_fec_sda_ok', 'h_fec_sda_okt'
    ]
    columnas_fechas_sin_guion = [
        'h_fec_reg', 'h_fec_lld', 'h_res_fec', 'h_ult_cam_fec', 'h_fec_sda'
    ]

    for fecha in columnas_fechas:
        if fecha in df.columns:
            df[fecha] = pd.to_datetime(df[fecha], errors='coerce')

    for fecha in columnas_fechas_sin_guion:
        if fecha in df.columns:
            df[fecha] = pd.to_datetime(df[fecha], format='%Y%m%d', errors='coerce')

    df = df.drop(columns=['h_cod_reserva', 'h_codigop'], errors='ignore')
    df = df.dropna()

    return df

def remove_duplicated_columns(df: pd.DataFrame) -> pd.DataFrame:
    resultados = []
    for col1, col2 in combinations(df.columns, 2):
        iguales = (df[col1] == df[col2]) & ~(df[col1].isna() | df[col2].isna())
        conteo = iguales.sum()
        if conteo == df.shape[0]:
            resultados.append(col2)
    df = df.drop(columns=resultados, errors='ignore')
    return df

def cardinality(df: pd.DataFrame) -> pd.DataFrame:
    # Porcentaje acumulado que deseamos conservar. Es decir, la nueva categoría "Otro" contendrá como máximo el 10% de los valores
    umbral = 0.90
    # Seleccionar columnas
    cols = ['ID_Agencia', 'ID_Tipo_Habitacion']

    # Crear y conservar solo las versiones reducidas
    for col in cols:
        freqs = df[col].value_counts(normalize=True)
        top_cats = freqs.cumsum()[freqs.cumsum() <= umbral].index
        df[col + '_reducida'] = df[col].astype(str).where(df[col].isin(top_cats), other='Otro')

    df.drop(columns=cols, inplace=True)
    df.columns = [col.replace('_reducida', '') for col in df.columns]
    return df

def exhaustive_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Quitar NAs
    df = df.dropna(axis=1, how='all')
    # Quitar zero columns
    df = df.loc[:, (df != 0).any(axis=0)]
    # Quitar mostly zero columns
    threshold = 0.98 * len(df)
    df = df.loc[:, (df == 0).sum() < threshold]
    # Descartar columnas sin variabilidad
    df = df.loc[:, df.nunique() > 1]

    # Quitar todas las variables de año anterior 
    cols_to_drop = [col for col in df.columns if col.startswith('aa_')]
    drop_columns_inplace(df, cols_to_drop)

    # Filtrar columnas iguales
    #resultados_iguales = comparar_columnas(df)
    drop_columns_inplace(df, ['Cliente_Disp'])

    # Quitar columnas sin variabilidad (> 99% valores iguales)
    drop_columns_inplace(df, ['ID_Pais_Origen', 'h_tot_hab'])

    # Corrección de categorías: Fechas 
    # Quitar diferentes formatos de la misma fecha
    df = df.loc[:, ~df.columns.str.contains('_ok|_okt|aa_')]

    # Convertir columnas de fecha a datetime
    fec_cols = [col for col in df.columns if '_fec' in col]
    convert_columns_type(df, fec_cols, 'datetime')

    # Verificar que las fechas de entrada y salida no sean inconsistentes
    assert ((df['h_fec_lld'] > df['h_fec_sda']).sum() == 0), "Hay fechas de check-in posteriores a las de check-out."

    # Corrección de categorías: Inconsistencias numéricas
    # Corregir inconsistencias de h_num_noc
    df.loc[df['h_num_noc'], 'h_num_noc'] = df['h_fec_sda'] - df['h_fec_lld']
    # Convertir h_num_noc a enteros y no datetime
    df['h_num_noc'] = df['h_num_noc'].apply( # type: ignore
        lambda x: x.days if isinstance(x, pd.Timedelta) else int(re.search(r'\d+', str(x)).group()) if re.search(r'\d+', str(x)) else None
    )
    # Hacer que el número de personas sea la suma de adultos y menores
    df['h_num_per'] = df['h_num_men'] + df['h_num_adu']

    # Mantener solo tarifas positivas
    df = df[df['h_tfa_total'] > 0]

    # Normalizar tarifa total
    df['tfa_xnoche'] = (df['h_tfa_total'] / df['h_num_noc'])
    drop_columns_inplace(df, ['tarifa_x_noche'])

    # Quitar outliers de tarifa por noche
    df = df[remove_outliers_iqr(df['tfa_xnoche'])]

    # Quitar grandes outliers de h_num_noc: no estadías de más de 30 días
    df = df[df['h_num_noc'] <= 30]

    # Mutación de columnas 
    # Agregar días de la semana y meses del año
    date_extraction(df, 'h_fec_reg', 'dia', 'reservacion')
    date_extraction(df, 'h_fec_lld', 'dia', 'entrada')
    date_extraction(df, 'h_fec_sda', 'dia', 'salida')
    date_extraction(df, 'h_fec_reg', 'mes', 'reservacion')
    date_extraction(df, 'h_fec_lld', 'mes', 'entrada')
    date_extraction(df, 'h_fec_sda', 'mes', 'salida')

    # Agregar días de anticipación de reserva
    df['dias_anticipacion'] = (df['h_fec_lld'] - df['h_fec_reg']).dt.days.clip(lower=0)

    # Agregar binaria de cancelación
    df['cancelado'] = np.where((df['ID_estatus_reservaciones'] == 2), 1, 0)

    return df

def prep_for_model(df: pd.DataFrame) -> pd.DataFrame:
    # Preparación para modelo
    # Eliminar columnas que no se consideran necesarias
    drop_columns_inplace(df, ['ID_canal', 'ID_estatus_reservaciones', 'Fecha_hoy', 'h_ult_cam_fec'])

    # Definir columnas numéricas y categóricas
    numeric_as_category = [col for col in df.columns if 'ID_' in col] + ['cancelado']
    numerical_cols, categorical_cols = get_numeric_and_categorical_columns(df, numeric_as_category)

    # Convertir a numéricas y categóricas
    convert_columns_type(df, numerical_cols, 'numeric')
    convert_columns_type(df, categorical_cols, 'categorical')

    # Eliminar filas con NAs
    df.dropna(inplace=True)

    # Crear nuevo dataset para el modelo
    df_model = df[numerical_cols + categorical_cols]

    # Quitar columnas no relevantes para el modelo
    drop_columns_inplace(df_model, ['ID_Reserva', 'h_res_fec', 'h_fec_lld', 'h_fec_sda', 'h_status', 'ID_estatus', 'entre/fin_reservacion'])

    # Quitar columnas que dificultan interpretabilidad
    drop_columns_inplace(df_model, ['h_cod_age', 'h_can_res', 'ID_Segmento_Comp'])

    # Actualizar las columnas numéricas y categóricas	
    numerical_cols, categorical_cols = get_numeric_and_categorical_columns(df_model)

    # Quitar columnas con colinealidad
    drop_columns_inplace(df_model, ['ID_Agencia', 'h_edo', 'dia_salida', 'mes_salida'])

    # Actualizar columnas numéricas
    _, categorical_cols = get_numeric_and_categorical_columns(df_model)

    # Identificar columnas binarias entre las categóricas
    binary_cols = []
    for col in categorical_cols[:]:  # [:] para iterar sobre una copia
        if df_model[col].nunique() == 2:
            binary_cols.append(col)
            categorical_cols.remove(col)

    # Preparación para modelo numérico
    df_num = df_model[numerical_cols + categorical_cols + binary_cols]

    # Preparación de variables cíclicas
    # Ciclicidad de días de la semana
    day_map = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }

    df_num['dia_reservacion'] = df_num['dia_reservacion'].map(day_map).astype(int)
    # df_num['dia_reservacion_sin'] = np.sin(2 * np.pi * df_num['dia_reservacion'] / 7)
    # df_num['dia_reservacion_cos'] = np.cos(2 * np.pi * df_num['dia_reservacion'] / 7)
    # drop_columns_inplace(df_num, ['dia_reservacion'])
    df_num['dia_entrada'] = df_num['dia_entrada'].map(day_map).astype(int)
    # df_num['dia_entrada_sin'] = np.sin(2 * np.pi * df_num['dia_entrada'] / 7)
    # df_num['dia_entrada_cos'] = np.cos(2 * np.pi * df_num['dia_entrada'] / 7)
    # drop_columns_inplace(df_num, ['dia_entrada'])

    # Ciclicidad de meses del año
    mes_map = {
        'January': 0, 'February': 1, 'March': 2, 'April': 3,
        'May': 4, 'June': 5, 'July': 6, 'August': 7,
        'September': 8, 'October': 9, 'November': 10, 'December': 11
    }

    df_num['mes_reservacion'] = df_num['mes_reservacion'].map(mes_map)
    # df_num['mes_reservacion_sin'] = np.sin(2 * np.pi * (df_num['mes_reservacion'].astype('category').cat.codes + 1) / 12)
    # df_num['mes_reservacion_cos'] = np.cos(2 * np.pi * (df_num['mes_reservacion'].astype('category').cat.codes + 1) / 12)
    # drop_columns_inplace(df_num, ['mes_reservacion'])
    df_num['mes_entrada'] = df_num['mes_entrada'].map(mes_map)
    # df_num['mes_entrada_sin'] = np.sin(2 * np.pi * (df_num['mes_entrada'].astype('category').cat.codes + 1) / 12)
    # df_num['mes_entrada_cos'] = np.cos(2 * np.pi * (df_num['mes_entrada'].astype('category').cat.codes + 1) / 12)
    # drop_columns_inplace(df_num, ['mes_entrada'])

    return df_num

def scale_and_encode(df_num: pd.DataFrame) -> pd.DataFrame:
    # Actualizar columnas categóricas
    _, categorical_cols = get_numeric_and_categorical_columns(df_num)

    # Preprocesamiento: Escalado y codificación
    preprocessor = ColumnTransformer([
        #("cyc_mtres", StandardScaler(), ['mes_reservacion_sin', 'mes_reservacion_cos']),
        #("cyc_dtres", StandardScaler(), ['dia_reservacion_sin', 'dia_reservacion_cos']),
        #("cyc_mtlld", StandardScaler(), ['mes_entrada_sin', 'mes_entrada_cos']),
        #("cyc_dtlld", StandardScaler(), []), #['dia_entrada_sin', 'dia_entrada_cos']
        ("num", StandardScaler(), ['h_num_adu', 'h_num_men', 'h_num_noc', 'tfa_xnoche']), #
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ['mes_reservacion', 'mes_entrada']), # 'ID_Paquete', 'ID_Tipo_Habitacion'
        #("bin", "passthrough", ['cancelado']) 
        #("bin", OneHotEncoder(drop='if_binary', sparse_output=False), binary_cols)
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessor)
    ])

    # Ajustar el pipeline al DataFrame
    X_transformed = pipeline.fit_transform(df_num)

    return X_transformed


def find_pca_elbow(X_transformed: pd.DataFrame) -> int:
    pca = PCA()
    pca.fit(X_transformed)
    individual_var = pca.explained_variance_ratio_

    # Calcular el índice del codo
    codo = encontrar_codo(individual_var)

    return codo


def apply_pca(X_transformed: pd.DataFrame, codo: int) -> np.ndarray:

    # Reducción de dimensionalidad
    pca = PCA(n_components=codo, random_state=42)
    X_pca = pca.fit_transform(X_transformed)

    return X_pca

