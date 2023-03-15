import re
import numpy as np
import os
from pandas import DataFrame


def extract(filename):
    with open(filename,'r') as f:
        lines=f.readlines()
    # assert "last" in lines[-84]
    epoch = None
    seed = None
    method = None
    rob = None
    eta2_flag = False
    for line in lines:
        epoch_obj = re.match(r'.*attack_epoch=(.*)$', line.strip())
        if epoch_obj is not None:
            epoch = int(epoch_obj.group(1))+1
            continue
        seed_obj = re.match(r'.*seed=(.*)$', line.strip())
        if seed_obj is not None:
            seed = int(seed_obj.group(1))
            continue
        if "Accuracy under attack" in line:
            clean_obj=re.search(r'(\d+(\.\d+)?)',line.strip())
            rob = float(clean_obj.group())
            continue
        method_obj = re.match(r'.*training_type=(.*)$', line.strip())
        if method_obj is not None:
            method = str(method_obj.group(1))
        if "eta2=0.0" not in line and "eta2" in line:
            eta2_flag = True
    # if method == "freelbvirtual":
    #     if eta2_flag:
    #         method = 'AccumWP'
    #     else:
    #         method = 'FreeLB-AWP-10'
    # elif method == 'weightpgd':
    #     method = 'PGD-AWP-10'
    # elif method == 'freelb':
    #     method = 'FreeLB'
    return epoch,seed,rob,method

if __name__ == "__main__":
    data_list = {
        "Epochs": [],
        "seed": [],
        "Method": [],
        "Accuracy under Attack (%)":[],
    }
    file_dir = [
        "manylogs/augmentratiobertattackseed21att.logs"
        # "manylogs/augmentratiobertattackdattepo.logs"
        # "sst2smalllargerepochattackepochs.logs", 
        # "manylogs/sst2smalllargerepochseedepochsaccumwpatt.logs",
        # "manylogs/sst2smalllargerepochseedepochspgdawpfreelb.logs"
    ]
    task_range = [range(0, 70)]
    for j in range(len(task_range)):
        for i in task_range[j]:
            epoch, seed, rob, method = extract(os.path.join(file_dir[j], f"task-{i}.txt"))
            data_list["Epochs"].append(epoch)
            data_list["seed"].append(seed)
            data_list["Method"].append(method)
            data_list["Accuracy under Attack (%)"].append(rob)
    df = DataFrame(data_list)
    df.to_csv('zha.csv')
    print(data_list)