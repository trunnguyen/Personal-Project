import logging
import os
import json
from datetime import datetime

def get_logger(log_dir: str = 'logs', name: str = 'DogCatCNN') -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'{name}_{timestamp}.log')

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"Add log to {log_path}")
    return logger

def save_history(history: dict, log_dir: str = 'logs', name: str = 'History'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(log_dir, f'{name}_{timestamp}.json')
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)
    return path