import os
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).resolve().parent[2]
LOG_DIR= os.path.join(PROJECT_ROOT,"data","job_history.log")

os.makedirs(os.path.dirname(LOG_DIR),exist_ok=True)

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_DIR,mode='a',encoding='utf-8'),
                              logging.StreamHandler()
                              ]
                    )
logger = logging.getLogger(__name__)
