import pandas as pd
from scipy import stats
import math
from sklearn.linear_model import LinearRegression
import networkx as nx
from Helper import print_dag
from Helper import para_pri
from Helper import parents
from Helper import detect_dag
import pickle
import numpy as np
from scipy.stats import multivariate_normal


def calculate_f_D_m(df, edge_list):
    with open("./data_files/epsilon.pickle", "rb") as f:
        epsilon_dict = pickle.load(f)
    res = 1
    pa_dict = parents(edge_list)

    for v in df.keys():
        mu_pri_list = []
        sigma_pri_list = []
        for pa in pa_dict[v]:
            para_pri_list = para_pri([pa, v])
            mu_pri_list.append(para_pri_list[0])
            sigma_pri_list.append(para_pri_list[1])
        mu_pri = np.array(mu_pri_list)
        Sigma_pri = np.diag(sigma_pri_list)

        Sigma = np.diag([epsilon_dict[v][1]]*len(df)) + np.dot(np.dot(df[pa_dict[v]].values, Sigma_pri), df[pa_dict[v]].values.T)
        mu = (np.dot(df[pa_dict[v]].values, mu_pri)+epsilon_dict[v][0]).reshape(len(df), 1)

        value = multivariate_normal.pdf(df[[v]].T.values[0].tolist(), mean=mu.T[0].tolist(), cov=Sigma.tolist())
        if value == 0:
            return None
        else:
            res *= value
    return res


def ges_algo(df):
    import random
    res_list = []
    # Initialize
    for v in df.keys():
        res_list.append([v])
    res = calculate_f_D_m(df, res_list)

    for v in df.keys():
        opt_pa = []
        for w in df.keys():
            if v == w:
                continue
            else:
                target_pa = opt_pa + [w]
                for pa in target_pa:
                    res_list.append([pa, v])
                target_res = calculate_f_D_m(df, res_list)
                try:
                    if target_res < res:
                        for pa in target_pa:
                            res_list.remove([pa, v])
                    else:
                        res = target_res
                # res or target_res should be None
                except TypeError:
                    return None
    # remove useless nodes to clean up
    pa_dict = parents(res_list)
    for key, value in pa_dict.items():
        if len(value) > 0:
            res_list.remove([key])
    print("*****Predicted CPDAG*****")
    print_dag(res_list)
    for v in df.keys():
        try:
            pa_list = parents(res_list)
            for w in pa_list[v]:
                if v in pa_list[w]:
                    # decide the arrow randomly

                    if random.randint(0, 1) == 1:
                        res_list.remove([v, w])
                        # if it's not dag, remove the edge
                        if detect_dag(res_list):
                            res_list.append([v, w])
                            res_list.remove([w, v])
                        if detect_dag(res_list):
                            res_list.remove([v, w])
                    else:
                        res_list.remove([w, v])
                        if detect_dag(res_list):
                            res_list.append([w, v])
                            res_list.remove([v, w])
                        if detect_dag(res_list):
                            res_list.remove([w, v])
        except KeyError:
            pass
    for v in df.keys():
        flag = 0
        for edge in res_list:
            if v in edge:
                flag = 1
        if flag == 0:
            res_list.append([v])
    print("*****Predicted DAG*****")
    print_dag(res_list)
    return res_list


def search_covariate(edge_list):
    DAG = nx.DiGraph()
    for edge in edge_list:
        nx.add_path(DAG, edge)
    UNDAG = DAG.to_undirected()
    pa_dict = parents(edge_list)
    X_pa_list = pa_dict['X']
    Y_pa_list = pa_dict['Y']
    ZZ = []
    ZZ_on_path = []
    if 'X' in ZZ:
        ZZ.remove('X')
    for path in nx.all_simple_paths(UNDAG, source='X', target='Y'):
        if path[1] in X_pa_list and path[-2] in Y_pa_list:
            ZZ.append(path[-2])
#             ZZ.append(path[-2])
        else:
            ZZ_on_path.append(path[-2])

    # append pa of Y not on path
    for pa in Y_pa_list:
        if pa not in ZZ_on_path:
            ZZ.append(pa)
    return ZZ


def ges_ate(df, edge_list, x_do):
    # if ['X', 'Y'] not in edge_list:
    #     return 0
    ZZ = ['X'] + search_covariate(edge_list)
    ZZ = sorted(set(ZZ), key=ZZ.index)
    clf = LinearRegression()
    X = df[ZZ].values
    Y = df['Y'].values
    clf.fit(X, Y)
    print(pd.DataFrame({"Name": df[ZZ].columns,
                        "Coefficients": clf.coef_}).sort_values(by='Coefficients'))
    print(clf.intercept_)
    return clf.coef_[0] * x_do


def ges_method(df):
    print("========GES========\n===================")
    # read x_do
    with open("./data_files/intervation_x.txt", "r") as f:
        x_do = float(f.read())
    predicted_model = ges_algo(df)
    if predicted_model is None:
        return None
    predicted_ate = ges_ate(df, predicted_model, x_do)
    print("*****Predicted ATE of GES Algorithm (x={0})*****\n{1}".format(x_do, predicted_ate))
    return predicted_ate
