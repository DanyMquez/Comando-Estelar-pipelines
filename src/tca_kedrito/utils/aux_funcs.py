import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
from itertools import combinations

# Eliminar columnas específicas
def drop_columns_inplace(df, columns):
    existing_cols = [col for col in columns if col in df.columns]
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        print(f"Algunas columnas no existen en la base de datos: {missing_cols}")
    if existing_cols:
        df.drop(columns=existing_cols, inplace=True)

# Carga del dataset con limpieza general
def load_dataset(df, **kwargs):
    try:
        drop_columns_inplace(df, ['h_nom', 'h_correo_e', 'h_codigop', 'cluster'])
    except Exception as e:
        print(f"Error: {e}")
        return None

    # Convertir espacios a NAs
    df = df.replace(r'^\s*$', np.nan, regex=True)
    # Quitar NAs 
    df = df.dropna(axis=1, how='all')
    # Quitar zero columns
    df = df.loc[:, (df != 0).any(axis=0)]
    # Quitar mostly zero columns
    threshold = 0.98 * len(df)
    df = df.loc[:, (df == 0).sum() < threshold]
    # Descartar columnas sin variabilidad
    df = df.loc[:, df.nunique() > 1]

    return df


# Identificar columnas numéricas y categóricas
def get_numeric_and_categorical_columns(df: pd.DataFrame, numeric_as_category=None) -> tuple:
    if numeric_as_category is None:
        numeric_as_category = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(numeric_as_category).tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.union(numeric_as_category).tolist()
    return numeric_cols, categorical_cols

# Extraer día o mes de una columna de fechas
def date_extraction(df, columna_fecha, tipo, sufijo):
    if columna_fecha not in df.columns:
        print(f"La columna '{columna_fecha}' no existe en el DataFrame.")
        return

    if tipo == 'dia':
        df[f'dia_{sufijo}'] = df[columna_fecha].dt.day_name()
    elif tipo == 'mes':
        df[f'mes_{sufijo}'] = df[columna_fecha].dt.month_name()
    else:
        print("El parámetro 'tipo' debe ser 'dia' o 'mes'.")


# Convertir columnas a tipos de datos específicos
def convert_columns_type(df, columns, dtype):
    for col in columns:
        if col not in df.columns:
            #print(f"Columna '{col}' no encontrada en el DataFrame.")
            continue
        try:
            if dtype == 'datetime':
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif dtype == 'numeric':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif dtype == 'categorical':
                df[col] = df[col].astype('category')
            else:
                raise ValueError("El tipo de dato debe ser 'datetime', 'numeric' o 'categorical'")
        except Exception as e:
            print(f"Error al convertir la columna '{col}': {e}")

# Eliminar outliers usando el método IQR
def remove_outliers_iqr(series, factor=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return series.between(lower, upper)

# Comparar columnas para encontrar duplicados
def comparar_columnas(df):
    resultados = []
    for col1, col2 in combinations(df.columns, 2):
        iguales = (df[col1] == df[col2]) & ~(df[col1].isna() | df[col2].isna())
        conteo = iguales.sum()
        resultados.append({
            'columna_1': col1,
            'columna_2': col2,
            'valores_iguales': conteo
        })
    return pd.DataFrame(resultados)

# Correlación de variables categóricas
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1)*(r - 1))/(n - 1))
    rcorr = r - ((r - 1)**2)/(n - 1)
    kcorr = k - ((k - 1)**2)/(n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

# Matriz de Cramér's V para variables categóricas
def cramers_v_matrix(df_cat):
    cols = df_cat.columns
    matrix = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                matrix.loc[col1, col2] = 1
            else:
                matrix.loc[col1, col2] = cramers_v(df_cat[col1], df_cat[col2])
    return matrix

# Encontrar el "codo" para PCA
def encontrar_codo(varianza):
    # Crear los puntos (x, y)
    puntos = np.vstack((range(1, len(varianza)+1), varianza)).T

    # Línea entre primer y último punto
    inicio, fin = puntos[0], puntos[-1]
    linea_vec = fin - inicio
    linea_vec = linea_vec / np.linalg.norm(linea_vec)

    # Calcular distancias perpendiculares desde cada punto a la línea
    distancias = []
    for punto in puntos:
        vec = punto - inicio
        proy = np.dot(vec, linea_vec) * linea_vec
        ort = vec - proy
        distancias.append(np.linalg.norm(ort))

    # Índice del máximo
    codo_idx = int(np.argmax(distancias)) + 1  # +1 porque los componentes empiezan en 1
    return codo_idx
