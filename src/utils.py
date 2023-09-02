import os
import sys
import dill #O objetivo principal do dill é serializar objetos Python mais complexos, como funções, classes definidas pelo usuário, instâncias de classes, entre outros,
            #que o pickle padrão pode não ser capaz de lidar.

import numpy as np
import pandas as pd
from src.exceptions import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

