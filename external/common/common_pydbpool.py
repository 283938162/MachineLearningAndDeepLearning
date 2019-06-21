# -*- coding: utf-8 -*-
import pymysql
import pymssql
import time

from DBUtils.PooledDB import PooledDB

"""
功能说明: 公共模块-数据库连接池模块
作者日期: 刘盛杰|2018年8月19日16:30:54
"""

# 声明全局变量
DBTYPE = 'mysql'


# 数据库连接配置信息字典
mysqlInfo = {
    "host": '192.168.5.222',
    "user": 'root',
    "passwd": '000000',
    "dbname": 'ROSAS',
    "port": 3306,
    "charset": 'utf8'
}

sqlServerInfo = {
    "host": '120.25.238.73',
    "user": 'web',
    "passwd": 'pass@word1',
    "dbname": 'ROSAS_HN',
    "charset": 'utf8'
}


class PyDBPool:
    """
    数据库连接池处理类
    """
    __pool = None

    def __init__(self, dbtype=DBTYPE) -> None:
        """
        构造函数
        :param dbtype: 数据库类型
        """
        self.conn = PyDBPool.getDBConn(self, dbtype.lower())
        self.cursor = self.conn.cursor()

    @staticmethod
    def getDBConn(self, dbtype):
        """
         获取数据库连接
        :param dbtype: 数据库类型
        :return:
        """
        if dbtype == 'mysql':
            if PyDBPool.__pool is None:
                __pool = PooledDB(creator=pymysql, mincached=1, maxcached=20, host=mysqlInfo['host'],
                                  user=mysqlInfo['user'], passwd=mysqlInfo['passwd'], db=mysqlInfo['dbname'],
                                  port=mysqlInfo['port'], charset=mysqlInfo['charset'])
                print("Create Mysql database connection pool succeed")
                return __pool.connection()
        elif dbtype == 'mssql':
            if PyDBPool.__pool is None:
                __pool = PooledDB(creator=pymssql, mincached=1, maxcached=20, host=sqlServerInfo['host'],
                                  user=sqlServerInfo['user'], password=sqlServerInfo['passwd'],
                                  database=sqlServerInfo['dbname'], charset=sqlServerInfo['charset'])
                print("Create SQLserver database connection pool succeed")
                return __pool.connection()
        else:
            print('Please enter the correct database type! MySQL or MSSQL')

    def update(self, sql, detail='off'):
        """
        更新 操作
        :param sql: 操作sql
        :return: 操作成功返回True,操作失败返回False
        """
        if detail == 'on':
            print("update_sql = %s" % [sql])
        try:
            self.cursor.execute(sql.replace('None', 'null'))
            self.conn.commit()
            return True
        except Exception as e:
            print("更新过程异常:%s" % e)
            return False

    def delete(self, sql,detail = 'off'):
        """
       删除 操作
       :param sql: 操作sql
       :return: 操作成功返回True,操作失败返回False
       """
        if detail == 'on':
            print("delete_sql = %s" % [sql])
        try:
            self.cursor.execute(sql.replace('None', 'null'))
            self.conn.commit()
            return True
        except Exception as e:
            print("删除过程异常:%s" % e)
            return False

    def insert(self, sql,detail='off'):
        """
       插入 操作
       :param sql: 操作sql
       :return: 操作成功返回True,操作失败返回False
       """
        if detail == 'on':
            print("insert_sql = %s" % [sql])
        try:
            self.cursor.execute(sql.replace('None', 'null'))
            self.conn.commit()
            return True
        except Exception as e:
            print("插入过程异常:%s" % e)
            return False

    def insertBatch(self, batchList, tableName):
        """
        批量插入列表数据 列表形式-> [(),(),(),(),(),(),()]
        :param batchList: 嵌套列表
        :param tableName: 插入数据库的表明
        :return: 操作成功返回True,操作失败返回False
        """
        try:
            insertSql = "insert into %s values(%s)" % (tableName, ','.join(['%s' for n in range(len(batchList[0]))]))
            self.cursor.executemany(insertSql, batchList)
            self.conn.commit()
            return True
        except Exception as e:
            print("批量插入过程异常:%s" % e)
            # print(e)
            return False

    def timeSelectCal(select):
        """
        AOP注解 计算sql执行时间
        :param select: 函数名
        :return:
        """

        def wrapper(*args, **kwargs):
            starttime = time.time()
            result = select(*args, **kwargs)
            endtime = time.time()
            timeInterval = endtime - starttime
            print('查询耗时 = %s' % timeInterval)
            return result

        return wrapper

    # @timeSelectCal
    def select(self, sql,detail='off',fetch='all'):
        """
        数据库查询函数\n
        :param sql: 查询sql
        :param fetch: 取数据方式: 'one' 取单条数据; 'all' 取全部数据
        :return: 返回满足条件的数据集
        """
        if detail == 'on':
            print("count = %s;sql = %s" % (self.count(sql), [sql.strip('\n')]))
        try:
            self.cursor.execute(sql)
            if fetch == 'one':
                return self.cursor.fetchone()
            elif fetch == 'all':
                return self.cursor.fetchall()
        except Exception as e:
            print("查询sql语句异常:%s" % e)

    def count(self, sql):
        """统计满足条件的sql结果集的行数"""
        sql = sql + ";select @@rowcount as count;"
        try:
            self.cursor.execute(sql)
            self.cursor.nextset()
            res = self.cursor.fetchone()

            if res is None:
                return 0
            else:
                return res[0]
        except Exception as e:
            print("统计查询受影响行数出现异常:%s" % e)

    def exec(self, sql):
        """执行存储过程语句"""
        try:
            self.cursor.execute(sql)
            self.conn.commit()
        except Exception as e:
            print("执行存储过程异常:%s" % e)

    def close(self):
        """连接资源释放"""
        self.cursor.close()
        self.conn.close()


if __name__ == '__main__':
    dbpool = PyDBPool()

    #sql = "select * from DBTest"
    #print(dbpool.select(sql))
    sql1="select   * from manager_task_detail limit 2"
    print(dbpool.select(sql1))
    # todo
    dbpool.close()
