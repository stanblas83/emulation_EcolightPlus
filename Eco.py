# -*- coding: utf-8 -*-

'''

@author: Doua

'''



# ================================= only change these two ========================================
SEED = 31200

setting_memo = "one_run/m"


# first column: for train, second column: for spre_train
list_traffic_files = [
    [["osm_pt.rou.xml"], ["osm_pt.rou.xml"]]
]

list_model_name = [
                    "Dueling"
                   ]

# ================================= only change these two ========================================


import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow 
tensorflow.random.set_seed(SEED)
import json
import os
import traffic_light_dueling
import time
import OHANA
import pandas as pd

PATH_TO_CONF = "conf/"+setting_memo

# Noise Prediction model with data fusion:
ohana = OHANA.OHANA(pre_train = True, model_weights='noise_par/OHANA.h5',history='noise_par/FinalDataAfter_imputation5n.csv')

# -------------------------------------Prediction Model------------------------------------------------
   
sumulation = 'real'

for model_name in list_model_name:
    
    for traffic_file, traffic_file_pretrain in list_traffic_files:

        print("toto")
        dic_exp = json.load(open(os.path.join(PATH_TO_CONF, "exp.conf"), "r"))
        
        dic_exp["MODEL_NAME"] = model_name
        dic_exp["TRAFFIC_FILE"] = traffic_file
        dic_exp["TRAFFIC_FILE_PRETRAIN"] = traffic_file_pretrain
        
        if sumulation == "real":
            dic_exp["RUN_COUNTS"] = 4801140
        elif sumulation == "synthetic":
            dic_exp["RUN_COUNTS"] = 216000
            
        json.dump(dic_exp, open(os.path.join(PATH_TO_CONF, "exp.conf"), "w"), indent=4)

        # change MIN_ACTION_TIME correspondingly

        dic_sumo = json.load(open(os.path.join(PATH_TO_CONF, "sumo_agent.conf"), "r"))
        if model_name == "Dueling":
            dic_sumo["MIN_ACTION_TIME"] = 2
        else:
            dic_sumo["MIN_ACTION_TIME"] = 1
        json.dump(dic_sumo, open(os.path.join(PATH_TO_CONF, "sumo_agent.conf"), "w"), indent=4)
        
        
        prefix = "{0}_{1}_{2}_{3}".format(
            dic_exp["MODEL_NAME"],
            dic_exp["TRAFFIC_FILE"],
            dic_exp["TRAFFIC_FILE_PRETRAIN"],
            time.strftime('%m_%d_%H_%M_%S_', time.localtime(time.time())) + "seed_%d" % SEED
        )

        traffic_light_dueling.main(memo=setting_memo, f_prefix=prefix, noise_model=ohana)

        print("finished {0}".format(traffic_file))
    print ("finished {0}".format(model_name))