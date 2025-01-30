from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
import h5py

def focal_loss_multi_class(alpha, gamma=2.0):
    alpha = tf.constant(alpha, dtype=tf.float32)

    def focal_loss_fixed(y_true, y_pred):
        # Clip predictions
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Standard cross-entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate alpha for the specific class
        alpha_factor = tf.reduce_sum(alpha * y_true, axis=1)

        # Probability of the true class
        p_t = tf.reduce_sum(y_pred * y_true, axis=1)

        # Focal factor
        focal_factor = tf.pow((1.0 - p_t), gamma)

        # Combine everything
        loss = alpha_factor * focal_factor * tf.reduce_sum(cross_entropy, axis=1)
        return tf.reduce_mean(loss)

    return focal_loss_fixed

# If you need class weights, define them here
class_weights = [0.25]

model_path = "modelo_final_bueno.h5"

try:
    with h5py.File(model_path, "r") as f:
        print("Modelo H5 cargado correctamente.")

    # Provide *all possible* loss names that could be stored in the .h5
    # so we avoid the "Unknown loss function" error.
    model = load_model(
        model_path,
        custom_objects={
            # The literal "loss" key if Keras saved it under 'loss'
            "loss": focal_loss_multi_class(alpha=class_weights),
            # The inner function name, if it was saved as focal_loss_fixed
            "focal_loss_fixed": focal_loss_multi_class(alpha=class_weights),
            # The outer function name, if it was saved as focal_loss_multi_class
            "focal_loss_multi_class": focal_loss_multi_class
        }
    )
    print("Modelo cargado exitosamente.")

except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Pydantic model for inputs
class ModelInput(BaseModel):
    last_pymnt_d_day_of_week: float
    last_pymnt_d_month: float
    last_credit_pull_d_month: float
    loan_age_years: float
    installment: float
    recoveries: float
    days_since_last_payment: float
    last_credit_pull_d_day_of_week: float
    out_prncp_inv: float
    out_prncp: float
    last_pymnt_d_year: float
    collection_recovery_fee: float
    issue_d_year: float
    last_pymnt_amnt: float
    last_credit_pull_d_year: float
    total_rec_prncp: float
    days_since_last_credit_pull: float
    total_pymnt: float
    total_pymnt_inv: float

@app.get("/")
def read_root():
    return {"message": "API para modelo de predicción"}

@app.post("/predict/")
def predict(input_data: ModelInput):
    if model is None:
        return {"error": "Modelo no cargado correctamente"}

    # Convertimos los datos en un array de numpy
    numeric_features = np.array([
        input_data.last_pymnt_d_day_of_week,
        input_data.last_pymnt_d_month,
        input_data.last_credit_pull_d_month,
        input_data.loan_age_years,
        input_data.installment,
        input_data.recoveries,
        input_data.days_since_last_payment,
        input_data.last_credit_pull_d_day_of_week,
        input_data.out_prncp_inv,
        input_data.out_prncp,
        input_data.last_pymnt_d_year,
        input_data.collection_recovery_fee,
        input_data.issue_d_year,
        input_data.last_pymnt_amnt,
        input_data.last_credit_pull_d_year,
        input_data.total_rec_prncp,
        input_data.days_since_last_credit_pull,
        input_data.total_pymnt,
        input_data.total_pymnt_inv
    ]).reshape(1, -1)

    # Hacer la predicción
    prediction = model.predict(numeric_features)

    # Si la salida es binaria (1 sola neurona en la capa de salida)
    if prediction.shape[1] == 1:
        prob_class_1 = float(prediction[0][0])
        prob_class_0 = 1.0 - prob_class_1
        return {
            "probability_class_0": prob_class_0,
            "probability_class_1": prob_class_1
        }

    # Si el modelo es multiclase (varias neuronas en la capa de salida)
    elif prediction.shape[1] > 1:
        class_probabilities = prediction[0].tolist()
        return {"class_probabilities": class_probabilities}

    return {"error": "El modelo no tiene un número de salidas esperado."}