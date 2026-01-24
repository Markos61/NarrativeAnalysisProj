from keras.models import load_model
from similarity_funcs import get_embedding
import numpy as np


def get_prediction(model_name, text):
    """Функция для получения предсказания """
    model = load_model(model_name)
    # Получаем embedding
    emb = get_embedding([text], verbose=False)  # например shape = (312,)
    emb = np.array(emb, dtype=np.float32)

    # Преобразуем в (batch, timesteps, features)
    if emb.ndim == 1:
        emb = np.expand_dims(emb, axis=0)  # batch
        emb = np.expand_dims(emb, axis=1)  # timesteps
    elif emb.ndim == 2:
        emb = np.expand_dims(emb, axis=0)  # batch

    pred = model.predict(emb)
    return pred
