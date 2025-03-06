from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.pipeline import Pipeline
import h5py
import joblib

# ---------------------------
# Función focal loss (para cargar el modelo)
def focal_loss_multi_class(alpha, gamma=2.0):
    alpha = tf.constant(alpha, dtype=tf.float32)
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        alpha_factor = tf.reduce_sum(alpha * y_true, axis=1)
        p_t = tf.reduce_sum(y_pred * y_true, axis=1)
        focal_factor = tf.pow((1.0 - p_t), gamma)
        loss = alpha_factor * focal_factor * tf.reduce_sum(cross_entropy, axis=1)
        return tf.reduce_mean(loss)
    return focal_loss_fixed

# ---------------------------
# Cargar el modelo Keras
model_path = "modelo_correccion1.h5"
class_weights = [0.25]  # Ajusta según corresponda

try:
    custom_objs = {
        "loss": focal_loss_multi_class(alpha=class_weights),
        "focal_loss_fixed": focal_loss_multi_class(alpha=class_weights),
        "focal_loss_multi_class": focal_loss_multi_class
    }
    with h5py.File(model_path, "r") as f:
        print("Archivo H5 abierto correctamente.")
    model = load_model(model_path, custom_objects=custom_objs)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

# ---------------------------
# Cargar el pipeline de preprocesamiento
try:
    pipeline = joblib.load('full_pipeline.pkl')
    print("Pipeline cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el pipeline: {e}")
    pipeline = None

# ---------------------------
# Crear la aplicación FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajusta según tus necesidades
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Endpoint GET /predict
# Recibe los parámetros en el siguiente orden:
# total_pymnt, installment, funded_amnt_inv, funded_amnt, int_rate,
# loan_amnt, tot_coll_amt, tot_cur_bal, total_rev_hi_lim, dti, revol_bal,
# revol_util, sub_grade
@app.get("/predict")
def predict(
    total_pymnt: float,
    installment: float,
    funded_amnt_inv: float,
    funded_amnt: float,
    int_rate: float,
    loan_amnt: float,
    tot_coll_amt: float,
    tot_cur_bal: float,
    total_rev_hi_lim: float,
    dti: float,
    revol_bal: float,
    revol_util: float,
    sub_grade: str
):
    if model is None or pipeline is None:
        return {"error": "El modelo o el pipeline no se cargaron correctamente."}
    
    # Construir un DataFrame con las variables en el orden requerido
    data = {
        "total_pymnt": [total_pymnt],
        "installment": [installment],
        "funded_amnt_inv": [funded_amnt_inv],
        "funded_amnt": [funded_amnt],
        "int_rate": [int_rate],
        "loan_amnt": [loan_amnt],
        "tot_coll_amt": [tot_coll_amt],
        "tot_cur_bal": [tot_cur_bal],
        "total_rev_hi_lim": [total_rev_hi_lim],
        "dti": [dti],
        "revol_bal": [revol_bal],
        "revol_util": [revol_util],
        "sub_grade": [sub_grade.strip()]  # quitar espacios en blanco
    }
    df_input = pd.DataFrame(data)
    
    # Transformar los datos con el pipeline
    try:
        X_final = pipeline.transform(df_input)
    except Exception as e:
        return {"error": f"Error en la transformación de datos: {e}"}
    
    # Realizar la predicción
    try:
        prediction = model.predict(X_final)
    except Exception as e:
        return {"error": f"Error en la predicción: {e}"}
    
    # Suponiendo que el modelo es multiclase:
    # Se obtiene el índice de la clase con mayor probabilidad y su valor
    if prediction.ndim == 2 and prediction.shape[1] > 1:
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        predicted_prob = float(np.max(prediction, axis=1)[0])
    else:
        # En caso de ser binario (una salida)
        predicted_class = 1 if prediction[0][0] >= 0.5 else 0
        predicted_prob = float(prediction[0][0]) if predicted_class == 1 else 1.0 - float(prediction[0][0])
    
    # Devolver como string "categoria probabilidad" separados por un espacio
    return f"{predicted_class} {predicted_prob}"

# ---------------------------
# Endpoint raíz
@app.get("/")
def read_root():
    return {"message": "API para modelo de predicción"}

# ---------------------------
# Ejecutar la aplicación (modo desarrollo)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
