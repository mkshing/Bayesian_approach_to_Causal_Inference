import networkx as nx
from IPython.display import SVG, display
import pickle
from random import randint
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import random


# pri is 0, 1/2
def random_pri():
    # return 0.5
    return random.uniform(-0.1, 0.1)


def random_var():
    return 1
    # return random.uniform(1, 1)


def print_dag(edge_list):
    DAG = nx.DiGraph()
    for edge in edge_list:
        nx.add_path(DAG, edge)

    svg = SVG(nx.nx_agraph.to_agraph(DAG).draw(prog='fdp', format='svg'))
    display(svg)


def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)


def detect_dag(edge_list):
    G = nx.DiGraph()
    for edge in edge_list:
        nx.add_path(G, edge)
    if nx.is_directed_acyclic_graph(G):
        return False
    else:
        return True


def random_dag(p, c):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
    vv = init_vv(p)
    zz = []
    for i in range(p):
        zz.append('Z_' + str(i + 1))
    # parent_prob = [0.19, 0.01]+[0.8/p]*p
    # decidde number of edges
    edges = randint(2, 2*p)
    # make a DAG
    G = nx.DiGraph()
    for v in vv:
        G.add_node(v)
    G.add_edge('X', 'Y')

    y_pa = c - 1
    while edges > 0:
        a = np.random.choice(vv)
        b = a
        while b == a:
            b = np.random.choice(vv)
        if a == 'X' and b == 'Y':
            pass
        # Z are indep
        elif a[0] == 'Z' and b[0] == 'Z':
            pass
        else:
            G.add_edge(a, b)
            if nx.is_directed_acyclic_graph(G):
                if b == 'Y' and y_pa > 0:
                    y_pa -= 0
                    edges -= 1
                elif b == 'Y':
                    G.remove_edge(a, b)
                else:
                    edges -= 1
            else:
                # we closed a loop!
                G.remove_edge(a, b)
    res_edge = [list(x) for x in list(G.edges)]
    # add nodes
    for edge in res_edge:
        for node in edge:
            if node in vv:
                vv.remove(node)
    for v in vv:
        res_edge.append([v])
    # print dag
    print_dag(res_edge)
    return res_edge

    # detect_c_res = detect_c(res_edge, c)
    # if detect_c_res[0]:
    #     # remove
    #     for z in zz:
    #         if [z, 'Y'] in res_edge and z not in detect_c_res[1]:
    #             res_edge.remove([z, 'Y'])
    #     # add nodes
    #     for edge in res_edge:
    #         for node in edge:
    #             if node in vv:
    #                 vv.remove(node)
    #     for v in vv:
    #         res_edge.append([v])
    #     # print dag
    #     print_dag(res_edge)
    #     return res_edge
    # else:
    #     return random_dag(p, c)


def return_kl(edge_list):
    l = []
    k = []

    DAG = nx.DiGraph()
    for edge in edge_list:
        nx.add_path(DAG, edge)
    pa_dict = parents(edge_list)
    for path in nx.all_simple_paths(DAG, source='X', target='Y'):
        if len(path) == 2:
            continue
        else:
            l.append(path[1])

    UNDAG = DAG.to_undirected()
    for path in nx.all_simple_paths(UNDAG, source='X', target='Y'):
        if path[1] in pa_dict['X'] and path[1] in pa_dict['Y']:
            k.append(path[1])

    print("-----k, l-----\n{}, {}".format(len(k), len(l)))
    print("k={}, l={}".format(k, l))
    return k, l


def para_pri(edge):
    with open("./data_files/prior_params.pickle", "rb") as f:
        pri_dict = pickle.load(f)
    param = [x['param'] for x in pri_dict if x['edge'] == edge]
    if len(param) == 1:
        return param[0]
    else:
        new_param = {'edge': edge, 'param': [random_pri(), random_var()]}
        pri_dict.append(new_param)
        # Update prior_params
        with open("./data_files/prior_params.pickle", "wb") as f:
            pickle.dump(pri_dict, f)
            print("Wrote prior_params in ./data_files/prior_params.pickle!")
        return new_param['param']


# Initialize vv
def init_vv(p):
    vv = []
    vv.append('X')
    vv.append('Y')
    for i in range(p):
        vv.append('Z_' + str(i + 1))
    return vv


def parents(edge_list):
    pa_dict = {}
    vv = []
    for e in edge_list:
        for node in e:
            if node not in vv:
                vv.append(node)
            edge_v = [x for x in edge_list if len(x) == 2 and x[1] == node]
            pa_v = [x[0] for x in edge_v]
            pa_dict[node] = pa_v
    return pa_dict


def squared_loss(true_ate, predicted_ate):
    loss = (true_ate - predicted_ate)**2
    print("*****Loss*****\n{}".format(loss))
    return loss


def set_dataframe(n, edge_list, params, write_bool=True):
    with open("./data_files/epsilon.pickle", "rb") as f:
        epsilon_dict = pickle.load(f)

    value_dict = {}
    G = nx.DiGraph()
    for edge in edge_list:
        nx.add_path(G, edge)
    data = []
    pa_dict = parents(edge_list)
    for i in range(n):
        for v in list(nx.topological_sort(G)):
            pa_v = pa_dict[v]
            theta_list = []
            for pa in pa_v:
                edge = [pa, v]
                theta_list.append("theta_" + "".join([node[2:] if node[0] == "Z" else node for node in edge]))
            mu_v = epsilon_dict[v][0]
            for theta, pa in zip(theta_list, pa_v):
                mu_v += params[theta]*value_dict[pa]
            value_dict[v] = np.random.normal(mu_v, epsilon_dict[v][1], 1)[0]
        row = {}
        for v in list(pa_dict.keys()):
            row[v] = value_dict[v]
        data.append(row)
    df = pd.DataFrame(data)
    if write_bool:
        # Write
        df.to_csv("./data_files/data.csv")
        print("Wrote dataframe in ./data_files/data.csv!")
    return df


def calculate_true_ate(edge_list, params, x_do):
    with open("./data_files/epsilon.pickle", "rb") as f:
        epsilon_dict = pickle.load(f)

    k, l = return_kl(edge_list)

    true_ate = params['theta_XY'] * x_do + epsilon_dict['Y'][0]

    for z in k:
        true_ate += params['theta_' + z[-1] + 'Y'] * epsilon_dict[z][0]

    for z_ in l:
        true_ate += (params['theta_' + 'X' + z_[-1]] * x_do + epsilon_dict[z_][0]) * params['theta_' + z_[-1] + 'Y']

    # G = nx.DiGraph()
    # for edge in edge_list:
    #     nx.add_path(G, edge)
    # paths = nx.all_simple_paths(G, source='X', target='Y')
    # for path in list(paths):
    #     theta_path = 1
    #     path = [node[2:] if node[0] == 'Z' else node for node in path]
    #     for i in range(len(path)-1):
    #         theta = "theta_" + path[i] + path[i+1]
    #         theta_path *= params[theta]
    #     true_ate += theta_path
    # true_ate *= x_do
    print("-------True ATE-------\n{}".format(true_ate))
    return true_ate


def simple_linear_reg(df):
    print("======LinearReg======\n===================")

    with open("./data_files/intervation_x.txt", "r") as f:
        x_do = float(f.read())
    clf = LinearRegression()

    X = df.loc[:, ['X']].values
    Y = df['Y'].values

    clf.fit(X, Y)
    predict_ate = clf.coef_[0] * x_do
    print(predict_ate)
    return predict_ate


def update_sigma_dict(p=100):
    vv = init_vv(p)
    sigma_dict = {}
    for v in vv:
        # scale should be more than 0
        # sigma_dict[v] = random.uniform(1, 2)
        sigma_dict[v] = random_var()
    # save the result
    with open("./data_files/sigma.pickle", "wb") as f:
        pickle.dump(sigma_dict, f)
        print("Wrote parameters in ./data_files/sigma.pickle!")


def update_epsilon_dict(p=100):
    vv = init_vv(p)
    epsilon_dict = {}
    for v in vv:
        # scale should be more than 0
        # random.uniform(-0.1, 0.1)
        # epsilon_dict[v] = [random_pri(), random_var()]
        epsilon_dict[v] = [3, 1]
    # save the result
    with open("./data_files/epsilon.pickle", "wb") as f:
        pickle.dump(epsilon_dict, f)
        print("Wrote parameters in ./data_files/epsilon.pickle!")

