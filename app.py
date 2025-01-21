from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

def focal_loss_multi_class(alpha, gamma=2.0):
    """
    alpha: array de tamaño [num_classes] con el peso para cada clase
    gamma: factor para penalizar más los ejemplos mal clasificados
    """
    alpha = tf.constant(alpha, dtype=tf.float32)

    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)

        alpha_factor = tf.reduce_sum(alpha * y_true, axis=1)
        p_t = tf.reduce_sum(y_pred * y_true, axis=1)
        focal_factor = tf.pow((1 - p_t), gamma)
        loss = alpha_factor * focal_factor * tf.reduce_sum(cross_entropy, axis=1)

        return tf.reduce_mean(loss)

    return focal_loss_fixed

class_weights = [0.25]
custom_loss = focal_loss_multi_class(alpha=class_weights)

model = load_model('modelo_final_local2.h5', custom_objects={'focal_loss_fixed': custom_loss})

app = FastAPI()

home_ownership_categories = ["ANY", "MORTGAGE", "NONE", "OTHER", "OWN", "RENT"]
verification_status_categories = ["Not Verified", "Source Verified", "Verified"]

home_ownership_mapping = {category: i for i, category in enumerate(home_ownership_categories)}
verification_status_mapping = {status: i for i, status in enumerate(verification_status_categories)}

class ModelInput(BaseModel):
    loan_amnt: float
    funded_amnt: float
    funded_amnt_inv: float
    int_rate: float
    installment: float
    annual_inc: float
    dti: float
    delinq_2yrs: float
    inq_last_6mths: float
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float
    total_acc: float
    total_pymnt: float
    total_pymnt_inv: float
    total_rec_prncp: float
    total_rec_int: float
    total_rec_late_fee: float
    last_pymnt_amnt: float
    collections_12_mths_ex_med: float
    acc_now_delinq: float
    tot_coll_amt: float
    tot_cur_bal: float
    total_rev_hi_lim: float
    emp_length_encoded: float
    term_encoded: float
    grade_encoded: float
    home_ownership_ANY: float
    home_ownership_MORTGAGE: float
    home_ownership_NONE: float
    home_ownership_OTHER: float
    home_ownership_OWN: float
    home_ownership_RENT: float
    verification_status_Not_Verified: float
    verification_status_Source_Verified: float
    verification_status_Verified: float

@app.get("/")
def read_root():
    return {"message": "API para modelo de predicción"}

@app.post("/predict/")
def predict(input_data: ModelInput):
    numeric_features = np.array([
        input_data.loan_amnt, input_data.funded_amnt, input_data.funded_amnt_inv, input_data.int_rate,
        input_data.installment, input_data.annual_inc, input_data.dti, input_data.delinq_2yrs,
        input_data.inq_last_6mths, input_data.open_acc, input_data.pub_rec, input_data.revol_bal,
        input_data.revol_util, input_data.total_acc, input_data.total_pymnt, input_data.total_pymnt_inv,
        input_data.total_rec_prncp, input_data.total_rec_int, input_data.total_rec_late_fee, input_data.last_pymnt_amnt,
        input_data.collections_12_mths_ex_med, input_data.acc_now_delinq, input_data.tot_coll_amt,
        input_data.tot_cur_bal, input_data.total_rev_hi_lim, input_data.emp_length_encoded, input_data.term_encoded,
        input_data.grade_encoded
    ])
    
    if not any([
        input_data.home_ownership_ANY, input_data.home_ownership_MORTGAGE, input_data.home_ownership_NONE,
        input_data.home_ownership_OTHER, input_data.home_ownership_OWN, input_data.home_ownership_RENT
    ]):
        return {"error": "Se requiere al menos una categoría de home_ownership codificada."}
    
    if not any([
        input_data.verification_status_Not_Verified, input_data.verification_status_Source_Verified,
        input_data.verification_status_Verified
    ]):
        return {"error": "Se requiere al menos una categoría de verification_status codificada."}

    data = np.concatenate((
        numeric_features, 
        np.array([
            input_data.home_ownership_ANY, input_data.home_ownership_MORTGAGE, input_data.home_ownership_NONE,
            input_data.home_ownership_OTHER, input_data.home_ownership_OWN, input_data.home_ownership_RENT,
            input_data.verification_status_Not_Verified, input_data.verification_status_Source_Verified,
            input_data.verification_status_Verified
        ])
    )).reshape(1, -1)

    prediction = model.predict(data)
    
    if prediction.shape[1] == 1:
        prob_class_1 = prediction[0][0]
        prob_class_0 = 1 - prob_class_1
        return {"probability_class_0": prob_class_0, "probability_class_1": prob_class_1}
    
    elif prediction.shape[1] > 1:
        class_probabilities = prediction[0].tolist()
        return {"class_probabilities": class_probabilities}
    
    return {"error": "El modelo no tiene un número de salidas esperado para clasificación binaria o multiclase"}
