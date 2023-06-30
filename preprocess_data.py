import pandas as pd
import numpy as np

def preprocess_data(csv_file):
    # Leer el archivo CSV de cada participante
    df = pd.read_csv(csv_file)
    
    # Remover las primeras 4 filas del CSV
    df = df.iloc[4:]
    
    # Tomar el primer valor de la columna "participant" y añadirlo al DataFrame
    participant_id = df['participant'].values[0]
    df['participant'] = participant_id
    
    # Conservar las columnas requeridas
    common_cols = ['participant', 'IAPS', 'Respuesta']
    cols1 = ['Respuesta_IAPs.keys', 'Respuesta_IAPs.rt']
    cols2 = ['Respuesta_prueba.keys', 'Respuesta_prueba.rt']
    
    if all(col in df.columns for col in cols1):
        selected_cols = common_cols + cols1
    elif all(col in df.columns for col in cols2):
        selected_cols = common_cols + cols2
    else:
        raise ValueError("Columnas requeridas no encontradas en el archivo CSV.")
    
    df = df[selected_cols]
    
    # Leer el archivo 'img_clsf.csv' que contiene el clasificador
    img_clsf_df = pd.read_csv('img_clsf.csv')

    # Verificar si los valores de la columna "IAPS" coinciden con alguna de las columnas relevantes
    if (df['IAPS'].values == img_clsf_df['HB1_img'].values).all():
        if 'HB1_clsf' in img_clsf_df.columns:
            df = df.merge(img_clsf_df[['HB1_img', 'HB1_clsf']], left_on='IAPS', right_on='HB1_img')
            df.drop(columns=['HB1_img'], inplace=True)
        else:
            raise ValueError("Columna 'HB1_clsf' no encontrada en el archivo 'img_clsf.csv'.")
    elif (df['IAPS'].values == img_clsf_df['HB2_img'].values).all():
        if 'HB2_clsf' in img_clsf_df.columns:
            df = df.merge(img_clsf_df[['HB2_img', 'HB2_clsf']], left_on='IAPS', right_on='HB2_img')
            df.drop(columns=['HB2_img'], inplace=True)
        else:
            raise ValueError("Columna 'HB2_clsf' no encontrada en el archivo 'img_clsf.csv'.")
    elif (df['IAPS'].values == img_clsf_df['MB1_img'].values).all():
        if 'MB1_clsf' in img_clsf_df.columns:
            df = df.merge(img_clsf_df[['MB1_img', 'MB1_clsf']], left_on='IAPS', right_on='MB1_img')
            df.drop(columns=['MB1_img'], inplace=True)
        else:
            raise ValueError("Columna 'MB1_clsf' no encontrada en el archivo 'img_clsf.csv'.")
    elif (df['IAPS'].values == img_clsf_df['MB2_img'].values).all():
        if 'MB2_clsf' in img_clsf_df.columns:
            df = df.merge(img_clsf_df[['MB2_img', 'MB2_clsf']], left_on='IAPS', right_on='MB2_img')
            df.drop(columns=['MB2_img'], inplace=True)
        else:
            raise ValueError("Columna 'MB2_clsf' no encontrada en el archivo 'img_clsf.csv'.")
    else:
        raise ValueError("Valores de la columna 'IAPS' no coinciden en el mismo orden con 'HB1_img', 'HB2_img', 'MB1_img' ni 'MB2_img'.")

    # Cambiar los nombres de las columnas
    df.columns = ['participant', 'img', 'exp_resp', 'act_resp', 'react_time', 'classifier']
    
    

    # Crear la columna 'result' con los resultados de comparación entre 'exp_resp' y 'act_resp'
    df['result'] = np.where((df['exp_resp'] == 'space') & (df['act_resp'] == 'space'), 'correct_answer',
                        np.where((df['exp_resp'].isna()) & (df['act_resp'].isna()), 'correct_inhibition',
                                 np.where((df['exp_resp'] == 'space') & (df['act_resp'].isna()), 'omission_error',
                                          np.where((df['exp_resp'].isna()) & (df['act_resp'] == 'space'), 'commission_error', 'unknown'))))

    # Crear la columna 'result_count' para contar los resultados
    df['result_count'] = np.where(df['result'] != 'unknown', 1, 0)

    return df
