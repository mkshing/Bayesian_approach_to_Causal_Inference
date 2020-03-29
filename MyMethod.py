import numpy as np
from random import randint
import networkx as nx
from Helper import para_pri
from Helper import parents
from Helper import init_vv
import pickle
from scipy.stats import multivariate_normal
import math


def bayes_method(df, c, rep_num, rep_kind=1):
    print("=========My Method========\n==========================")
    vv = list(df.keys())
    p = len(vv) - 2
    # read x_do
    with open("./data_files/intervation_x.txt", "r") as f:
        x_do = float(f.read())

    z_set = init_zset(p, c)
    zmin_set = init_zmin_set(p, c)

    print("~~~~Calculate all p(k, l| D)~~~~")
    all_p_k_D_dict = all_approx_p_k_D(df, zmin_set, z_set, rep_num, rep_kind)
    try:
        if all_p_k_D_dict is None:
            return None
    except Exception as e:
        print(e)
        pass
    bayes_ate = 0
    print("~~~~Calculate tau(x | k,l, D)~~~~")
    for k in zmin_set:
        for l in z_set[zmin_set.index(k)]:
            # print("======Searching in k={}======".format(k))
            tau_k_D = calculate_tau_k_D(df, x_do, k, l)
            # print("-----tau(x | {0}, D)-----\n{1}".format(k, tau_k_D))

            bayes_k = tau_k_D * [x['p_k_D'] for x in all_p_k_D_dict if x['k'] == k and x['l'] == l][0]

            bayes_ate += bayes_k

    print("******Bayes ATE******\n{}".format(bayes_ate))
    return bayes_ate


# return a list like [[Z_1], ['Z_2'], ..., ['Z_2', 'Z_3']]
def init_zmin_set(p=3, c=2):
    zmin_set = [[]]
    zz = []
    for i in range(p):
        zz.append('Z_' + str(i + 1))
    for z in zz:
        tmp = []
        for element in zmin_set:
            tmp.append(element + [z])
        zmin_set.extend(tmp)
    return [res for res in zmin_set if len(res) < c + 1]


def init_zset(p, c):
    zy_set = init_zmin_set(p, c)

    z_set = {}
    # initilize
    for i in range(len(zy_set)):
        z_set[i] = []

    for idx, zy in enumerate(zy_set):
        other_zz = ['Z_'+str(i+1) for i in range(p) if 'Z_'+str(i+1) not in zy]
        for zz in zy_set:
            flag = 0
            for z in zz:
                if z not in other_zz:
                    flag = 1
            if flag == 0 and len(zy)+len(zz) <= c:
                z_set[idx].append(zz)
    return z_set


# Initialize edge set
def init_edge(k=None, l=None, vv=None, print_bool=True):
    edge_k = []
    if vv == None:
        print("Input vv")
        return None

    if len(k) == 0 and len(l)==0:
        edge_k.append(['X', 'Y'])
        for v in vv:
            if v not in ['X', 'Y']:
                edge_k.append([v])
    else:
        edge_k.append(['X', 'Y'])
        for Z in k:
            edge_k.append([Z, 'X'])
            edge_k.append([Z, 'Y'])
        for Z_ in l:
            edge_k.append(['X', Z_])
            edge_k.append([Z_, 'Y'])
        for v in vv:
            if v in ['X', 'Y'] + k + l:
                pass
            else:
                edge_k.append([v])

    if print_bool:
        # print("-------Base model of k={}------".format(k))
        # print_dag(edge_k)
        pass
    return edge_k


# Using in approx_f_D_m
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


# mean of results
def approx_f_D_m(df, rep_models):
    res = 0
    for model in rep_models:
        tmp_res = calculate_f_D_m(df, model)
        if tmp_res is None:
            return None
        res += tmp_res
    res = res / len(rep_models)
    return res


# all_k_list is created by init_zmin_set()
def all_approx_p_k_D(df, all_k_list, all_l_list, rep_num, rep_kind):
    vv = list(df.keys())

    tmp_list = []
    # TODO: Set hyper-param. For now, it's uniform.
    # path = "./data_files/param-eta-c" + str(c) + ".pickle"
    # with open(path, "rb") as f:
    #     eta_set = pickle.load(f)
    count = 0
    for z in all_l_list.values():
        count += len(z)
    uni_prob = 1/count
    # Calculate each log_p_k_D
    for k in all_k_list:
        for l in all_l_list[all_k_list.index(k)]:
            # print("k={}, l={}".format(k, l))
            # rep_num < 3**(p-c) - 2*(p-c)
            rep_models_k = rep_models(k, l, vv, rep_num, rep_kind)
            approx_f_D_k = approx_f_D_m(df, rep_models_k)

            if approx_f_D_k is None:
                return None

            pk = uni_prob
            tmp_list.append(pk * approx_f_D_k)
    # denom
    # print("******denom******\n{}".format(sum(tmp_list)))
    denom = sum(tmp_list)

    res_list = []
    count = 0
    for k in all_k_list:
        for l in all_l_list[all_k_list.index(k)]:
            numerator = tmp_list[count]
            res = numerator / denom
            if np.isnan(res):
                return None
            p_k = {
                'k': k,
                'l': l,
                'p_k_D': res
            }
            res_list.append(p_k)
            count += 1

    # Return the dict of all sorted p(k | D)
    p_k_D_list = list(set([res['p_k_D'] for res in res_list]))
    sorted_res_list = []
    for p in sorted(p_k_D_list, reverse=True):
        tmp_list = [res for res in res_list if res['p_k_D'] == p]
        for d in tmp_list:
            sorted_res_list.append(d)

    print("***All p(k | D) <SORTED>***\n{}".format(sorted_res_list))
    return sorted_res_list


# Output repesented models for k
def rep_models(k=None, l=None, vv=None, rep_num=-1, rep_kind=1):
    # print("-------Finding the represented models for k={}".format(k))
    models = []
    p = len(vv) - 2
    if rep_num < 0:
        print("******Input rep_num******")
        rep_num = int(input())

    max_edges = int((2 * p - 2 * (len(k) + len(l))))

    # if rep_num doesn't seem to satisfy, rep_num will be less automatically
    if max_edges < 6:
        rep_num = 1 + (2 * p - 2 * (len(k) + len(l)))*2
        max_edges = 1 + (2 * p - 2 * (len(k) + len(l)))*2

    while rep_num > 0:
        # less edges
        if rep_kind == 1:
            # less than 50%
            edge_num = randint(1, max_edges * 0.5)
        # more edges
        elif rep_kind == 2:
            # more than 50%
            edge_num = randint(max_edges * 0.5, max_edges)
        elif rep_kind == 3:
            edge_num = randint(1, max_edges)
        else:
            print("Error on rep_kind")
            return None

        tmp_model = get_dag(p=p, k=k, l=l, edge_num=edge_num)
        if len(tmp_model) == 0:
            pass
        elif tmp_model not in models:
            models.append(tmp_model)
            rep_num -= 1
    return models


def calculate_param_pos(df, v, pa_dict):
    with open("./data_files/epsilon.pickle", "rb") as f:
        epsilon_dict = pickle.load(f)
    mu_pos_dict = {}

    mu_pri_list = []
    sigma_pri_list = []
    epsilon_mu = epsilon_dict[v][0]
    epsilon_sigma = epsilon_dict[v][1]
    zz = []
    for pa in pa_dict:
        if pa == 'X' and v[0] == 'Z':
            mu_pos_dict["theta_" + pa + v[-1]] = None
        elif pa == 'X':
            mu_pos_dict["theta_" + pa + v] = None
        else:
            zz.append(pa)
            mu_pos_dict["theta_" + pa[2:] + v] = None
        para_pri_list = para_pri([pa, 'Y'])
        mu_pri_list.append(para_pri_list[0])
        sigma_pri_list.append(para_pri_list[1])

    mu_pri = np.array(mu_pri_list)
    Sigma_pri = np.diag(sigma_pri_list)

    B = np.dot(df[pa_dict].values.T, df[pa_dict].values)
    B = B / epsilon_sigma
    Sigma_pos = np.linalg.inv((np.linalg.inv(Sigma_pri) + B))
    mu_pos = np.dot(Sigma_pos,
                    np.dot(df[pa_dict].values.T, df[['Y']].values - epsilon_mu) / epsilon_sigma + np.dot(Sigma_pri,
                                                                                                         mu_pri).reshape(
                        len(pa_dict), 1))

    for theta, mu in zip(mu_pos_dict.keys(), mu_pos):
        mu_pos_dict[theta] = mu[0]

    return mu_pos_dict


def calculate_tau_k_D(df, x_do, k, l):
    with open("./data_files/epsilon.pickle", "rb") as f:
        epsilon_dict = pickle.load(f)

    pa_dict = parents(init_edge(k=k, l=l, vv=list(df.keys())))

    res = epsilon_dict['Y'][0]
    mu_pos_y = calculate_param_pos(df, 'Y', pa_dict['Y'])
    res += mu_pos_y['theta_XY'] * x_do

    for z in k:
        res += mu_pos_y['theta_' + z[-1] + 'Y'] * epsilon_dict[z][0]

    for z_ in l:
        mu_pos_z = calculate_param_pos(df, z_, pa_dict[z_])
        res += (mu_pos_z['theta_' + 'X' + z_[-1]] * x_do + epsilon_dict[z_][0]) * mu_pos_y['theta_' + z_[-1] + 'Y']
    return res


def get_dag(p, k, l, edge_num):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
    vv = init_vv(p)
    zz = []
    for i in range(p):
        zz.append('Z_' + str(i + 1))

    # Set base model of k
    base_model = init_edge(k, l, vv)
    # make a DAG
    DAG = nx.DiGraph()
    for edge in base_model:
        nx.add_path(DAG, edge)

    # remove Z in k
    for node in k:
        zz.remove(node)
    for node in l:
        zz.remove(node)

    roop_count = 0
    while edge_num > 0:
        # if it takes double time of edges num, try another edge_num
        roop_count += 1
        if roop_count > 100:
            return []

        if edge_num > (2 * p - 2 * (len(k) + len(l))) * 0.8:
            a = np.random.choice(['X', 'Y'])
        else:
            a = np.random.choice(['X', 'Y'] + zz, p=[0.3] * 2 + [0.4 / len(zz)] * len(zz))
        b = a
        while b == a:
            if a[0] == 'Z':
                xy_list = [v for v in ['X', 'Y'] if not DAG.has_edge(a, v)]
                if len(xy_list) == 0:
                    xy_list = [a]
                b = np.random.choice(xy_list)
            else:
                b = np.random.choice(zz)
        if not DAG.has_edge(a, b):
            DAG.add_edge(a, b)
            if nx.is_directed_acyclic_graph(DAG) and detect_k(DAG, k, l):
                edge_num -= 1
            else:
                # we closed a loop!
                DAG.remove_edge(a, b)
        else:
            pass

    res_edge = [list(x) for x in list(DAG.edges)]
    # add nodes
    vv = init_vv(p)
    for edge in res_edge:
        for node in edge:
            if node in vv:
                vv.remove(node)
    for v in vv:
        res_edge.append([v])

    return res_edge


def detect_k(DAG, k, l):
    edge_list = [list(x) for x in list(DAG.edges)]
    pa_dict = parents(edge_list)

    UNDAG = DAG.to_undirected()
    for path in nx.all_simple_paths(UNDAG, source='X', target='Y'):
        if path[1] in k or path[1] in l:
            pass
        # include k
        elif path[1] in pa_dict['X'] and path[1] in pa_dict['Y']:
            return False
        # include l
        elif 'X' in pa_dict[path[1]] and path[1] in pa_dict['Y']:
            return False
    return True


def update_eta():
    import pickle
    # Set hyper-params
    eta_set_1 = [
        {'k': [], 'p_k': 0.25},
        {'k': ['Z_1'], 'p_k': 0.25},
        {'k': ['Z_2'], 'p_k': 0.25},
        {'k': ['Z_3'], 'p_k': 0.25},
    ]
    eta_set_2 = [
        {'k': [], 'p_k': 0.05},
        {'k': ['Z_1'], 'p_k': 0.05},
        {'k': ['Z_2'], 'p_k': 0.1},
        {'k': ['Z_3'], 'p_k': 0.2},
        {'k': ['Z_1', 'Z_2'], 'p_k': 0.1},
        {'k': ['Z_1', 'Z_3'], 'p_k': 0.2},
        {'k': ['Z_2', 'Z_3'], 'p_k': 0.3},
    ]
    eta_set_3 = [
        {'k': [], 'p_k': 0.05},
        {'k': ['Z_1'], 'p_k': 0.05},
        {'k': ['Z_2'], 'p_k': 0.05},
        {'k': ['Z_3'], 'p_k': 0.1},
        {'k': ['Z_1', 'Z_2'], 'p_k': 0.1},
        {'k': ['Z_1', 'Z_3'], 'p_k': 0.2},
        {'k': ['Z_2', 'Z_3'], 'p_k': 0.3},
        {'k': ['Z_1', 'Z_2', 'Z_3'], 'p_k': 0.15}
    ]
    for eta_set in [eta_set_1, eta_set_2, eta_set_3]:
        S = 0
        for k in eta_set:
            S += k['p_k']
        if round(S, 5) != 1:
            print("*******NOT SATISFY*********\n{}".format(eta_set))
            return eta_set
    with open("./data_files/param-eta-c1.pickle", "wb") as f1:
        pickle.dump(eta_set_1, f1)
        print("Wrote hyper-param eta in ./data_files/param-eta-c1.pickle!")
    with open("./data_files/param-eta-c2.pickle", "wb") as f2:
        pickle.dump(eta_set_2, f2)
        print("Wrote hyper-param eta in ./data_files/param-eta-c2.pickle!")
    with open("./data_files/param-eta-c3.pickle", "wb") as f3:
        pickle.dump(eta_set_3, f3)
        print("Wrote hyper-param eta in ./data_files/param-eta-c3.pickle!")



