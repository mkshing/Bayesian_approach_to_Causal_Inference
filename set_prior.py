import pickle
import numpy as np
from Helper import random_pri
from Helper import random_var
import random


def update_pri_params():
    # read hyper-parameters
    with open("./data_files/prior_params.pickle", "rb") as f:
        pri_dict = pickle.load(f)
    # update all params
    print("Updating all params")
    for x in pri_dict:
        # x['param'] = [random_pri(), random_var()]
        x['param'] = [random.uniform(-0.1, 0.1), 1/2]
    # write prior_params
    with open("./data_files/prior_params.pickle", "wb") as f:
        pickle.dump(pri_dict, f)
    print("Wrote prior_params in ./data_files/prior_params.pickle!")
    print("Updated all params!")


def set_pri_params(edge_list):
    try:
        # read hyper-parameters
        with open("data_files/prior_params.pickle", "rb") as f:
            pri_dict = pickle.load(f)
    except FileNotFoundError:
        pri_dict = []
    count = 0
    # check whethere it has a param for the edge
    for edge in edge_list:
        if len([x for x in pri_dict if x['edge'] == edge]) == 0 and len(edge) > 1:
            count += 1
            pri_dict.append({'edge': edge, 'param': [random_pri(), random_var()]})
    if count > 0:
        # write prior_params
        with open("./data_files/prior_params.pickle", "wb") as f:
            pickle.dump(pri_dict, f)
        print("Wrote prior_params in ./data_files/prior_params.pickle!")
    return pri_dict


def generate_params(pri_dict):
    # Set prior dists
    param_list = []
    theta_list = []
    for p in pri_dict:
        pri_value = np.random.normal(p['param'][0], p['param'][1], 1)[0]
        # print("prior of {}: {}".format(p['edge'], pri_value))
        theta_list.append("theta_" + "".join([node[2:] if node[0] == "Z" else node for node in p['edge']]))
        param_list.append(pri_value)

    # save the result as a text file
    with open("./data_files/parameters.pickle", "wb") as f:
        pickle.dump(param_list, f)
        print("Wrote parameters in ./data_files/parameters.pickle!")

    # Make a dict
    params = {}
    for param, k in zip(param_list, theta_list):
        params[k] = float(param)
    # print("-----Parameters----\n{}".format(params))
    return params
