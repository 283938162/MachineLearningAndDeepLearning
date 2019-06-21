import os
import sys
import time
import datetime

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import csv
from common.common_pydbpool import PyDBPool

# 全局变量
SAMPLE_MIN = 11
MIN_POINT = 10
MIN_DIST = 50



def calcu_location(location_lon, location_lat, r=50):
    lat_range = 180 / np.pi * r / 1000 / 6371.137
    lon_range = lat_range / np.cos(location_lat * np.pi / 180)
    max_lat = float('%.9f' % (location_lat + lat_range))
    min_lat = float('%.9f' % (location_lat - lat_range))
    max_lon = float('%.9f' % (location_lon + lon_range))
    min_lon = float('%.9f' % (location_lon - lon_range))
    range_xy = {}
    range_xy['location_lat'] = {'min': min(min_lat, max_lat), 'max': max(min_lat, max_lat)}
    range_xy['location_lon'] = {'min': min(min_lon, max_lon), 'max': max(min_lon, max_lon)}
    return range_xy



def eps_lonlat(x1, y1, r=50):
    range1 = calcu_location(x1, y1, r)
    x_dist = euclidean((y1, x1), (y1, range1['location_lon']['max']))
    y_dist = euclidean((y1, x1), (range1['location_lat']['max'], x1))
    print("x_dist = %s, y_dist = %s" % (x_dist, y_dist))
    return (x_dist + y_dist) / 2.0


def cal_cluster(narr, lonindex, latindex, min_point=10, min_dist=50):
    """
    计算聚类簇数 返回簇类label
    """
    X = np.stack([narr[:, lonindex], narr[:, latindex]], axis=-1)
    X = X.astype(np.float64)

    x1 = X[:, 0].mean()
    y1 = X[:, 1].mean()
    #print(x1, y1)
    eps_fix = eps_lonlat(x1, y1, min_dist)

    y_db = DBSCAN(eps=eps_fix, min_samples=min_point).fit(X)
    X_labels = y_db.labels_
    print('每个样本的簇标号:')
    print(X_labels)
    raito = len(X_labels[X_labels[:] == -1]) / len(X_labels)
    print('噪声比:', format(raito, '.2%'))
    n_clusters_ = len(set(X_labels)) - (1 if -1 in X_labels else 0)
    print('分簇的数目: %d' % n_clusters_)
    return X_labels


def area_auto_evaluate():
    dbpool = PyDBPool()
    sql1 = "SELECT sc.def_cellname_chinese FROM SCENE_INFO s JOIN SCENE_GRID_CELL_OTT sc ON s.DEF_CELLNAME = sc.DEF_CELLNAME WHERE s.SCENE_NAME = '南湖大学' AND sc.TTIME = '2019-01-01' AND sc.`STATUS`='1' AND sc.CELL_RSRP_AVG<=-110"
    sql2 = "SELECT sc.def_cellname_chinese,sc.grid_id, sc.grid_lon, sc.grid_lat FROM SCENE_INFO s JOIN SCENE_GRID_CELL_OTT sc ON s.DEF_CELLNAME = sc.DEF_CELLNAME WHERE s.SCENE_NAME = '南湖大学' AND sc.TTIME = '2019-01-01' AND sc.`STATUS`='1' AND sc.CELL_RSRP_AVG<=-110 and sc.def_cellname_chinese = '{0}'"
    cells = [cell[0] for cell in dbpool.select(sql1)]
    result = {}
    for cell in cells:
        result[cell] = dbpool.select(sql2.format(cell))
    for cellname, gridname in result.items():
        olist = np.array(gridname)
        if len(olist) < SAMPLE_MIN:
            print("样本数据过小，现为：%s " % len(olist))
            continue

        ## 数据聚类
        lonindex, latindex = (2, 3)
        X_labels = cal_cluster(olist, lonindex, latindex, MIN_POINT, MIN_DIST)
        df = pd.DataFrame(list(zip(gridname, X_labels)), index=None, columns=['栅格编号', '簇编号'])
        df.to_csv(r"D:\WorkSpace\PycharmProjects\MachineLearningAndDeepLearning\external\demo-{0}.csv".format(cellname), index=False)

    dbpool.close()


def area_issue_main():
    start1 = time.time()
    area_auto_evaluate()
    print("run cost:", time.time() - start1)


if __name__ == "__main__":
    area_issue_main()

