from datetime import datetime
import os
import sys

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from utils.files import save_json


def save_model(model, params, label_name):
    now = datetime.now()
    formatted_date = now.strftime("%d.%m")
    base_save_path = params['model_save_path']
    experimant_name = params['experimant_name']
    
    model_save_path = os.path.join(base_save_path, f"{experimant_name}_{formatted_date}", label_name)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # model.save_pretrained(model_save_path)
    save_json(params, os.path.join(model_save_path, 'params.json'))
    print(f"Params saved to {model_save_path}")
    return model_save_path



