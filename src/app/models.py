import joblib


# Function for load the model
def load_model(model):
    # Choice of model
    models = {
        "lr": "lr_model.pkl",
        "dt": "dt_model.pkl",
        "rf": "rf_model.pkl",
        "xgb": "xgb_model.pkl",
    }

    model_path = "models/"
    model_name = models[model]
    model = joblib.load(model_path + model_name)
    return model
