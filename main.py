from Helper import random_dag
from Helper import calculate_true_ate
from Helper import set_dataframe
from Helper import squared_loss
from Helper import simple_linear_reg
from Helper import update_epsilon_dict
from set_prior import set_pri_params
from set_prior import generate_params
from set_prior import update_pri_params
from MyMethod import bayes_method
from GES import ges_method
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import datetime
import logging
import random


logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename="test.log")
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter('%(asctime)s - %(funcName)s - %(lineno)s - %(message)s'))
logger.addHandler(handler)


def write_excel(rows, col_name, n):
    df_res = pd.DataFrame(rows, columns=col_name)

    # Write to csv file
    dt_now = datetime.datetime.now()
    sheet = "n=" + str(n) + "-" + dt_now.strftime('%Y-%m-%d-%H-%M-%S')
    path = "./data_files/loss_log.xlsx"
    if os.path.exists(path):
        with pd.ExcelWriter(path, engine="openpyxl", mode="a") as writer:
            df_res.to_excel(writer, sheet_name=sheet, index=False)
    else:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df_res.to_excel(writer, sheet_name=sheet, index=False)

    print("DONE!!\nWrite in the sheet={} of data_files/loss_log.xlsx".format(sheet))


def test(loop_num=5, n=10, p=7, c=4, rep_num=4, rep_kind=1, update_param=False, update_epsilon=False):
    if update_param:
        update_pri_params()
    if update_epsilon:
        update_epsilon_dict()

    lr_loss_list = []
    ges_loss_list = []
    bayes_loss_list = []
    for i in tqdm(range(loop_num)):
        lr_ate = None
        bayes_ate = None
        ges_ate = None
        while bayes_ate is None or ges_ate is None or lr_ate is None:
            dag_edge = random_dag(p, c)
            pri_dict = set_pri_params(dag_edge)
            params = generate_params(pri_dict)

            x_do = 5.0
            print("------Intervention------\nX = {}".format(str(x_do)))

            true_ate = calculate_true_ate(dag_edge, params, x_do)

            df = set_dataframe(n, dag_edge, params, write_bool=False)

            lr_ate = simple_linear_reg(df=df)
            ges_ate = ges_method(df=df)
            if ges_ate is None:
                continue
            bayes_ate = bayes_method(df=df, c=c, rep_num=rep_num, rep_kind=rep_kind)
        lr_loss = squared_loss(predicted_ate=lr_ate, true_ate=true_ate)
        ges_loss = squared_loss(predicted_ate=ges_ate, true_ate=true_ate)
        bayes_loss = squared_loss(true_ate=true_ate, predicted_ate=bayes_ate)
        # if lr_loss + 50 < bayes_loss:
        #     continue
        lr_loss_list.append(lr_loss)
        ges_loss_list.append(ges_loss)
        bayes_loss_list.append(bayes_loss)
        print("******MSE******\n{}, {}, {}".format(np.mean(lr_loss_list), np.mean(ges_loss_list),
                                                   np.mean(bayes_loss_list)))

    lr_res = np.mean(lr_loss_list)
    lr_std = np.std(lr_loss_list)
    ges_res = np.mean(ges_loss_list)
    ges_std = np.std(ges_loss_list)
    bayes_res = np.mean(bayes_loss_list)
    bayes_std = np.std(bayes_loss_list)
    print("*****MSE Of LinearReg*****\n{}".format(lr_res))
    print("*****MSE Of GES*****\n{}".format(ges_res))
    print("*****MSE Of Bayes*****\n{}".format(bayes_res))
    logger.debug("loop_num={}, n={}, p={}, c={}, rep_num={}, rep_kind={}".format(loop_num, n, p, c, rep_num, rep_kind))
    logger.debug("MSE Of LinearReg={}".format(lr_res))
    logger.debug("STD of LinearReg={}".format(lr_std))
    logger.debug("MSE Of GES={}".format(ges_res))
    logger.debug("STD of GES={}".format(ges_std))
    logger.debug("MSE Of Bayes={}".format(bayes_res))
    logger.debug("STD of Bayes={}".format(bayes_std))

    rows = []
    for lr, ges, bayes in zip(lr_loss_list, ges_loss_list, bayes_loss_list):
        rows.append([lr, ges, bayes])
    write_excel(rows=rows, col_name=['lr', 'GES', 'Bayes'], n=n)

    return [lr_res, ges_res, bayes_res, lr_std, ges_std, bayes_std]


def test_n(loop_num=1, p=7, c=4, rep_num=40, rep_kind=1):
    rows = []
    for n in tqdm([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]):
        res = test(loop_num=loop_num, n=n, p=p, c=c, rep_num=rep_num, rep_kind=rep_kind)
        rows.append([n, res[0], res[1], res[2], res[3], res[4], res[5]])

    df_res = pd.DataFrame(rows, columns=['n', 'MSE of LinearReg', 'MSE of GES', 'MSE of Bayes', 'STD of LinearReg',
                                         'STD of GES', 'STD of Bayes'])

    # Write to csv file
    dt_now = datetime.datetime.now()
    sheet = dt_now.strftime('%Y-%m-%d-%H-%M-%S')
    path = "./data_files/loss_result_n.xlsx"
    if os.path.exists(path):
        with pd.ExcelWriter(path, engine="openpyxl", mode="a") as writer:
            df_res.to_excel(writer, sheet_name=sheet, index=False)
    else:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df_res.to_excel(writer, sheet_name=sheet, index=False)

    print("DONE!!\nWrite in the sheet={} of data_files/loss_result_n.xlsx".format(sheet))


# [3, 10]
def test_p(loop_num=100, n=30, rep_num=40, rep_kind=1):
    rows = []
    for p in tqdm([3, 7]):
        res = test(loop_num=loop_num, n=n, p=p, c=round(p*0.6), rep_num=rep_num, rep_kind=rep_kind)
        rows.append([p, res[0], res[1], res[2]])

    df_res = pd.DataFrame(rows, columns=['p', 'MSE of LinearReg', 'MSE of GES', 'MSE of Bayes'])

    # Write to csv file
    dt_now = datetime.datetime.now()
    sheet = dt_now.strftime('%Y-%m-%d-%H-%M-%S')
    path = "./data_files/loss_result_p.xlsx"
    if os.path.exists(path):
        with pd.ExcelWriter(path, engine="openpyxl", mode="a") as writer:
            df_res.to_excel(writer, sheet_name=sheet, index=False)
    else:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df_res.to_excel(writer, sheet_name=sheet, index=False)

    print("DONE!!\nWrite in the sheet={} of data_files/loss_result_p.xlsx".format(sheet))


# [2, 7]
def test_c(loop_num=100, n=30, p=7, rep_num=40, rep_kind=1):
    rows = []
    for c in tqdm([2, 7]):
        res = test(loop_num=loop_num, n=n, p=p, c=c, rep_num=rep_num, rep_kind=rep_kind)
        rows.append([c, res[0], res[1], res[2]])

    df_res = pd.DataFrame(rows, columns=['c', 'MSE of LinearReg', 'MSE of GES', 'MSE of Bayes'])

    # Write to csv file
    dt_now = datetime.datetime.now()
    sheet = dt_now.strftime('%Y-%m-%d-%H-%M-%S')
    path = "./data_files/loss_result_c.xlsx"
    if os.path.exists(path):
        with pd.ExcelWriter(path, engine="openpyxl", mode="a") as writer:
            df_res.to_excel(writer, sheet_name=sheet, index=False)
    else:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df_res.to_excel(writer, sheet_name=sheet, index=False)

    print("DONE!!\nWrite in the sheet={} of data_files/loss_result_c.xlsx".format(sheet))


# [1, 3]
def test_c_mine(loop_num=100, n=30, p=7, rep_num=40, rep_kind=1):
    c_list = [1, 3]
    ges_loss_list = []
    lr_loss_list = []
    bayes_loss_list = {}

    for c in c_list:
        bayes_loss_list[c] = []

    for i in tqdm(range(loop_num)):
        lr_ate = None
        ges_ate = None
        bayes_ate = {}
        for c in c_list:
            bayes_ate[c] = None

        while all([v is None for v in bayes_ate.values()]) or ges_ate is None or lr_ate is None:
            dag_edge = random_dag(p, p)
            pri_dict = set_pri_params(dag_edge)
            params = generate_params(pri_dict)
            x_do = 5.0
            print("------Intervention------\nX = {}".format(str(x_do)))
            true_ate = calculate_true_ate(dag_edge, params, x_do)
            df = set_dataframe(n, dag_edge, params, write_bool=False)

            lr_ate = simple_linear_reg(df=df)
            ges_ate = ges_method(df=df)
            for c in tqdm(c_list):
                bayes_ate[c] = bayes_method(df=df, c=c, rep_num=rep_num, rep_kind=rep_kind)

        lr_loss_list.append(squared_loss(true_ate=true_ate, predicted_ate=lr_ate))
        ges_loss_list.append(squared_loss(true_ate=true_ate, predicted_ate=ges_ate))
        for c in c_list:
            bayes_loss_list[c].append(squared_loss(true_ate=true_ate, predicted_ate=bayes_ate[c]))

    row = [np.mean(lr_loss_list), np.mean(ges_loss_list)]
    colomn = ['MSE of LinearReg', 'MSE of GES']
    for c in c_list:
        row.append(np.mean(bayes_loss_list[c]))
        colomn.append('MSE of Bayes on c={}'.format(c))

    df_res = pd.DataFrame([row], columns=colomn)
    # Write to csv file
    dt_now = datetime.datetime.now()
    sheet = dt_now.strftime('%Y-%m-%d-%H-%M-%S')
    path = "./data_files/loss_result_c.xlsx"
    if os.path.exists(path):
        with pd.ExcelWriter(path, engine="openpyxl", mode="a") as writer:
            df_res.to_excel(writer, sheet_name=sheet, index=False)
    else:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df_res.to_excel(writer, sheet_name=sheet, index=False)
    print("DONE!!\nWrite in the sheet={} of data_files/loss_result_c.xlsx".format(sheet))


# [5, 10, 15]
def test_rep_num(loop_num=100, n=10, p=5, c=4, rep_kind=1):
    logger.debug("loop_num={}, n={}, p={}, c={}, rep_kind={}".format(loop_num, n, p, c, rep_kind))
    rep_num_list = [5, 10, 15]
    ges_loss_list = []
    lr_loss_list = []
    bayes_loss_list = {}

    for rep_num in rep_num_list:
        bayes_loss_list[rep_num] = []

    for i in tqdm(range(loop_num)):
        lr_ate = None
        ges_ate = None
        bayes_ate = {}
        for rep_num in rep_num_list:
            bayes_ate[rep_num] = None

        while all([v is None for v in bayes_ate.values()]) or ges_ate is None or lr_ate is None:
            dag_edge = random_dag(p, c)
            pri_dict = set_pri_params(dag_edge)
            params = generate_params(pri_dict)
            x_do = 5.0
            print("------Intervention------\nX = {}".format(str(x_do)))
            true_ate = calculate_true_ate(dag_edge, params, x_do)
            df = set_dataframe(n, dag_edge, params, write_bool=False)

            lr_ate = simple_linear_reg(df=df)
            ges_ate = ges_method(df=df)
            for rep_num in tqdm(rep_num_list):
                bayes_ate[rep_num] = bayes_method(df=df, c=c, rep_num=rep_num, rep_kind=rep_kind)

        lr_loss_list.append(squared_loss(true_ate=true_ate, predicted_ate=lr_ate))
        ges_loss_list.append(squared_loss(true_ate=true_ate, predicted_ate=ges_ate))
        for rep_num in rep_num_list:
            bayes_loss_list[rep_num].append(squared_loss(true_ate=true_ate, predicted_ate=bayes_ate[rep_num]))

    row = [np.mean(lr_loss_list), np.mean(ges_loss_list)]
    colomn = ['MSE of LinearReg', 'MSE of GES']
    for rep_num in rep_num_list:
        row.append(np.mean(bayes_loss_list[rep_num]))
        colomn.append('MSE of Bayes on rep_num={}'.format(rep_num))

    df_res = pd.DataFrame([row], columns=colomn)
    # Write to csv file
    dt_now = datetime.datetime.now()
    sheet = dt_now.strftime('%Y-%m-%d-%H-%M-%S')
    path = "./data_files/loss_result_rep_num.xlsx"
    if os.path.exists(path):
        with pd.ExcelWriter(path, engine="openpyxl", mode="a") as writer:
            df_res.to_excel(writer, sheet_name=sheet, index=False)
    else:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df_res.to_excel(writer, sheet_name=sheet, index=False)
    print("DONE!!\nWrite in the sheet={} of data_files/loss_result_rep_num.xlsx".format(sheet))


def test_rep_kind(loop_num=100, n=10, p=5, c=4, rep_num=10):
    logger.debug("loop_num={}, n={}, p={}, c={}, rep_num={}".format(loop_num, n, p, c, rep_num))

    lr_loss_list = []
    ges_loss_list = []
    bayes_loss_list = {}

    for rep_kind in [1, 2, 3]:
        bayes_loss_list[rep_kind] = []

    for i in tqdm(range(loop_num)):
        lr_ate = None
        ges_ate = None
        bayes_ate = {}
        for rep_kind in [1, 2, 3]:
            bayes_ate[rep_kind] = None

        while all([v is None for v in bayes_ate.values()]) or ges_ate is None or lr_ate is None:
            dag_edge = random_dag(p, c)
            pri_dict = set_pri_params(dag_edge)
            params = generate_params(pri_dict)
            x_do = 5.0
            print("------Intervention------\nX = {}".format(str(x_do)))
            true_ate = calculate_true_ate(dag_edge, params, x_do)
            df = set_dataframe(n, dag_edge, params, write_bool=False)

            lr_ate = simple_linear_reg(df=df)
            ges_ate = ges_method(df=df)
            for rep_kind in tqdm([1, 2, 3]):
                bayes_ate[rep_kind] = bayes_method(df=df, c=c, rep_num=rep_num, rep_kind=rep_kind)

        lr_loss_list.append(squared_loss(true_ate=true_ate, predicted_ate=lr_ate))
        ges_loss_list.append(squared_loss(true_ate=true_ate, predicted_ate=ges_ate))
        for rep_kind in [1, 2, 3]:
            bayes_loss_list[rep_kind].append(squared_loss(true_ate=true_ate, predicted_ate=bayes_ate[rep_kind]))

    row = [np.mean(lr_loss_list), np.mean(ges_loss_list)]
    colomn = ['MSE of LinearReg', 'MSE of GES']
    for rep_kind in [1, 2, 3]:
        row.append(np.mean(bayes_loss_list[rep_kind]))
        colomn.append('MSE of Bayes on rep_kind={}'.format(rep_kind))

    df_res = pd.DataFrame([row], columns=colomn)
    # Write to csv file
    dt_now = datetime.datetime.now()
    sheet = dt_now.strftime('%Y-%m-%d-%H-%M-%S')
    path = "./data_files/loss_result_rep_kind.xlsx"
    if os.path.exists(path):
        with pd.ExcelWriter(path, engine="openpyxl", mode="a") as writer:
            df_res.to_excel(writer, sheet_name=sheet, index=False)
    else:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df_res.to_excel(writer, sheet_name=sheet, index=False)
    print("DONE!!\nWrite in the sheet={} of data_files/loss_result_rep_kind.xlsx".format(sheet))


def test_ges_vs_lr(loop_num=1, n=30, p=7, c=4):
    # update_pri_params()
    lr_loss_list = []
    ges_loss_list = []
    count = 0
    for i in tqdm(range(loop_num)):
        lr_ate = None
        ges_ate = None
        while ges_ate is None or lr_ate is None:
            count += 1
            dag_edge = random_dag(p, c)
            pri_dict = set_pri_params(dag_edge)
            params = generate_params(pri_dict)

            x_do = 5.0
            print("------Intervention------\nX = {}".format(str(x_do)))
            true_ate = calculate_true_ate(dag_edge, params, x_do)

            df = set_dataframe(n, dag_edge, params, write_bool=False)
            lr_ate = simple_linear_reg(df=df)
            ges_ate = ges_method(df=df)
        lr_loss = squared_loss(predicted_ate=lr_ate, true_ate=true_ate)
        ges_loss = squared_loss(predicted_ate=ges_ate, true_ate=true_ate)

        lr_loss_list.append(lr_loss)
        ges_loss_list.append(ges_loss)

    lr_res = np.mean(lr_loss_list)
    ges_res = np.mean(ges_loss_list)
    print("*****MSE Of LinearReg*****\n{}".format(lr_res))
    print("*****MSE Of GES*****\n{}".format(ges_res))
    print(count)
    return [lr_res, ges_res]


def update(update_param=False, update_epsilon=False):
    if update_param:
        update_pri_params()
    if update_epsilon:
        update_epsilon_dict()
    print("Updated!")


if __name__ == '__main__':
    # update(update_param=True, update_epsilon=True)
    # rep_num < 3**(p-c) - 2*(p-c)
    # test(loop_num=10, n=5, p=5, c=3, rep_num=10, rep_kind=3, update_param=False, update_epsilon=False)

    # test_n(loop_num=100, p=5, c=3, rep_num=10, rep_kind=3)
    # test_p(loop_num=100, n=10, rep_num=10, rep_kind=3)
    # test_c(loop_num=100, n=10, p=5, rep_num=10, rep_kind=1)
    test_c_mine(loop_num=100, n=10, p=5, rep_num=10, rep_kind=3)
    # test_rep_kind(loop_num=100, n=10, p=5, c=3, rep_num=10)
    test_rep_num(loop_num=100, n=10, p=5, c=3, rep_kind=3)
    # test_ges_vs_lr(loop_num=10, n=10, p=5, c=3)
