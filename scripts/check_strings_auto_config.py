#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
组串自动配置检查脚本
该脚本用于在外部服务器上使用crontab定时任务运行，自动校正系统组串接入数据。
所有配置都在脚本内部，独立运行，不依赖项目其他模块。

执行步骤：
1. 显示菜单选择执行场景
2. 根据场景查询逆变器列表
3. 遍历逆变器列表，逐个处理组串配置
4. 更新sys_inverter_string_config表
5. 逆变器Redis缓存清除

支持场景：
- 1. 只查询没有配置组串记录的逆变器
- 2. 只查询零电流告警组串的逆变器
- 3. 按逆变器SN查询并配置
- 4. 按电站名称查询所有逆变器并配置
- 5. 按租户名称查询所有逆变器并配置
- 6. 查询1+2 (没有配置+零电流告警)

场景说明：
场景1：原组串接入配置不为空
- 子场景1a：逆变器SN在整改记录表(modRecord.csv)中
  - 最新数据高电流组串数量<整改后组串数量 →  使用最新数据配置且保留与原组串配置一致的组串以保证与整改后组串数量相等并记录日志
  - 最新数据高电流组串数量=整改后组串数量 → 使用最新数据配置并记录日志
  - 最新数据高电流组串数量>整改后组串数量 → 使用最新数据配置并删除该SN记录并记录日志
- 子场景1b：原组串接入配置不为空，原配置、最新、最旧数据三者数量一致但高电流组串顺序不一致 → 使用最新数据配置并记录日志
- 子场景1c：原组串接入配置不为空，原配置、最新、最旧数据三者数量一致且高电流组串顺序一致 → 保持原配置不需要记录日志
- 子场景1d：原组串接入配置不为空，且最新和最旧数据都为空 → 检查support_string，为0则全设置为接入并记录日志，不为0则保持原配置不需要记录日志
- 子场景1e：原组串接入配置不为空，最新数据为空，最旧数据不为空，且最旧数据高电流组串数量与原配置一致 → 保持原配置不需要记录日志
- 子场景1f：原组串接入配置不为空，最新数据为空，最旧数据不为空，且最旧数据高电流组串数量与原配置不一致 → 使用高电流组串数量最多的进行配置并记录日志
- 子场景1g：原组串接入配置不为空，最旧数据为空，最新数据不为空，且最新数据高电流组串数量与原配置一致 → 需要比较最新数据高电流组串顺序与原配置是否一致，一致则保持原配置，不一致则使用最新数据配置并记录日志
- 子场景1h：原组串接入配置不为空，最旧数据为空，最新数据不为空，且最新数据高电流组串数量与原配置不一致 → 使用高电流组串数量最多的进行配置并记录日志
- 子场景1i：原组串接入配置不为空，原配置、最新、最旧数据三者高电流组串数量都不相等或其中一个大于另外两个时 → 使用高电流组串数量最多的进行配置并记录日志
- 子场景1j：原组串接入配置不为空，原配置与最新数据高电流组串数量一致且大于最旧数据 →  需要比较最新数据高电流组串顺序与原配置是否一致，一致则保持原配置，不一致则使用最新数据配置并记录日志
- 子场景1k：原组串接入配置不为空，原配置与最旧数据高电流组串数量一致且大于最新数据 →   保持原配置不需要记录日志
场景2：原组串接入配置为空
- 子场景2a：最新和最旧数据不为空且一致 → 直接使用最新数据配置不需要记录日志
- 子场景2b：最新和最旧数据都为空 → 检查support_string，为0则全设置为接入并记录日志，不为0则跳过并记录日志
- 子场景2c：最新和最旧数据不为空但不一致 → 使用高电流组串数量最多的最新/最旧数据进行配置并记录日志
- 子场景2d：最新/最旧数据有一个为空一个不为空 → 直接使用不为空的有高电流组串的最新/最旧数据进行配置不需要记录日志
"""

import sys
import os
import csv
import json
import mysql.connector
from pymongo import MongoClient
import redis
from datetime import datetime, timedelta

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

LOG_FILE = os.path.join(project_root, "string_auto.log")

# DB_CONFIG = {
#     "host": "192.168.180.200",
#     "port": 32306,
#     "user": "root",
#     "password": "xxxxx",
#     "database": "ems"
# }

# MONGODB_CONFIG = {
#     "host": "192.168.180.200",
#     "port": 32017,
#     "username": "root",
#     "password": "xxxxx",
#     "database": "iot_device"
# }

# MONGODB_COOL_CONFIG = {
#     "host": "192.168.180.200",
#     "port": 32017,
#     "username": "root",
#     "password": "xxxx",
#     "database": "iot_device"
# }

# REDIS_CONFIG = {
#     "host": "192.168.180.200",
#     "port": 32379,
#     "password": "xxxx",
#     "db": 0
# }

DB_CONFIG = {
    "host": "172.28.81.107",
    "port": 3306,
    "user": "root",
    "password": "xxxxx",
    "database": "ems"
}

MONGODB_CONFIG = {
    "host": "172.28.81.107",
    "port": 27017,
    "username": "root",
    "password": "xxxxx",
    "database": "iot_device"
}

MONGODB_COOL_CONFIG = {
    "host": "172.28.81.106",
    "port": 27778,
    "username": "root",
    "password": "xxxxx",
    "database": "iot_device"
}

REDIS_CONFIG = {
    "host": "172.28.81.107",
    "port": 6379,
    "password": "xxxxx",
    "db": 0
}


CURRENT_THRESHOLD = 2 #电流阈值
POWER_RATIO = 0.1  #有功功率阈值占比
GENERATION_POWER_RATIO = 0.1 #发电量阈值占逆变器额定功率的比例，1.0=额定功率(默认), 0.5=额定功率的50%
DEBUG_MODE = False  #是否开启调试模式
RECENT_MONTHS_COUNT = "all"  # 0=只查并网当月+次月, 1-12=并网当月+次月+N个最近月, "all"=并网日期到上月所有月
MANUAL_CONFIRM = False  # True=新/旧数据有一个为空时需手动选择, False=使用默认逻辑


def get_mongodb_connection_string(config):
    from urllib.parse import quote_plus
    username = quote_plus(config["username"])
    password = quote_plus(config["password"])
    host = config["host"]
    port = config["port"]
    database = config["database"]
    return f"mongodb://{username}:{password}@{host}:{port}/{database}?authSource=admin&directConnection=true"


def log_support_string(plant_name, inverter_sn, support_string, action, tenant_name=""):
    update_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    reason = "最新和最旧数据都为空"
    
    log_entry = f"""时间：{update_time}
租户名称：{tenant_name}
电站名称：{plant_name}
逆变器SN:{inverter_sn}
组串配置更新结果：{action}
原因：{reason}

"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)


def log_mod_record(plant_name, inverter_sn, mod_count, newest_count, action):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {plant_name}--{inverter_sn}\n")
        f.write(f"整改后组串数量:{mod_count}, 最新数据高电流组串数量:{newest_count}\n")
        f.write(f"{action}\n")
        f.write("\n")


def _pad_label(label, width=26):
    """
    将标签填充到指定显示宽度（中文字符按2列计算）
    """
    display_width = sum(2 if ord(c) > 127 else 1 for c in label)
    padding = max(0, width - display_width)
    return label + ' ' * padding


def log_auto_config(plant_name, inverter_sn, action, detail=""):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {plant_name}--{inverter_sn}\n")
        f.write(f"{action}\n")
        if detail:
            f.write(f"详情: {detail}\n")
        f.write("\n")


MOD_RECORD_FILE = os.path.join(project_root, "modRecord.csv")


def read_mod_record():
    mod_record = {}
    if os.path.exists(MOD_RECORD_FILE):
        encodings = ["utf-8-sig", "gbk", "gb18030", "utf-8"]
        for enc in encodings:
            try:
                with open(MOD_RECORD_FILE, "r", encoding=enc) as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        if len(row) >= 2:
                            sn = row[0].strip()
                            count = row[1].strip()
                            if sn and count.isdigit():
                                mod_record[sn] = int(count)
                print(f"整改记录表读取成功 (编码: {enc})")
                return mod_record
            except UnicodeDecodeError:
                continue
            except Exception as e:
                msg = f"读取整改记录表失败: {str(e)}"
                print(msg)
                log_auto_config("SYSTEM", "N/A", msg)
                sys.exit(1)
        msg = f"读取整改记录表失败: 所有编码({encodings})均无法解析文件"
        print(msg)
        log_auto_config("SYSTEM", "N/A", msg)
        sys.exit(1)
    else:
        print(f"整改记录表不存在: {MOD_RECORD_FILE}")
    return mod_record


def delete_mod_record(inverter_sn):
    if not os.path.exists(MOD_RECORD_FILE):
        return

    rows = []
    encodings = ["utf-8-sig", "gbk", "gb18030", "utf-8"]
    for enc in encodings:
        try:
            with open(MOD_RECORD_FILE, "r", encoding=enc) as f:
                reader = csv.reader(f)
                header = next(reader)
                rows.append(header)
                for row in reader:
                    if len(row) >= 1 and row[0].strip() != inverter_sn:
                        rows.append(row)
            with open(MOD_RECORD_FILE, "w", encoding="utf-8-sig", newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            print(f"已从整改记录表中删除SN: {inverter_sn}")
            return
        except UnicodeDecodeError:
            continue
        except Exception as e:
            msg = f"删除整改记录失败: {str(e)}"
            print(msg)
            log_auto_config("SYSTEM", inverter_sn, msg)
            sys.exit(1)
    msg = f"删除整改记录失败: 所有编码({encodings})均无法解析文件"
    print(msg)
    log_auto_config("SYSTEM", inverter_sn, msg)
    sys.exit(1)


class CheckStringsAutoConfig:
    def __init__(self):
        self.db_config = DB_CONFIG

    def _get_mysql_connection(self):
        return mysql.connector.connect(**self.db_config)

    def query_inverters_without_config(self):
        """
        查询没有配置组串记录的inverter_sn
        """
        query = """
        SELECT 
            p.inverter_sn, 
            p.project_id, 
            p.plant_id, 
            sp.plant_name, 
            sp.tenant_id, 
            st.tenant_name,
            sp.grid_connection_date 
        FROM sys_inverter_project p 
        JOIN sys_plant sp 
          ON p.plant_id = sp.plant_id 
        JOIN sys_tenant st 
          ON sp.tenant_id = st.id 
         AND st.is_del = '0'        
        WHERE p.is_del = '0' 
          AND NOT EXISTS ( 
              SELECT 1 
              FROM sys_inverter_string_config c 
              WHERE c.project_id = p.project_id 
                AND c.status = '1' 
          ) 
        ORDER BY st.tenant_name, sp.plant_name, p.inverter_sn
        """

        conn = None
        cursor = None
        try:
            conn = self._get_mysql_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query)
            results = cursor.fetchall()
            print(f"查询到 {len(results)} 个未配置组串的逆变器")
            return results
        except Exception as e:
            print(f"查询未配置组串的逆变器时出错: {str(e)}\nSQL语句:\n{query}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def query_inverters_with_zero_current_alarm(self):
        """
        查询零电流告警组串的逆变器SN
        """
        query = """
        SELECT 
            p.inverter_sn, 
            p.project_id, 
            p.plant_id, 
            sp.plant_name, 
            sp.tenant_id, 
            st.tenant_name, 
            sp.grid_connection_date,
            a.alarm_name 
        FROM sys_inverter_project p 
        JOIN sys_plant sp 
          ON p.plant_id = sp.plant_id 
        JOIN sys_tenant st 
          ON sp.tenant_id = st.id 
        JOIN iot_device_alarm a 
          ON p.inverter_sn = a.device_no 
        WHERE p.is_del = '0'
          AND a.alarm_code = '101002' 
          AND a.status = '1' 
        ORDER BY st.tenant_name
        """

        conn = None
        cursor = None
        try:
            conn = self._get_mysql_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query)
            results = cursor.fetchall()
            print(f"查询到 {len(results)} 个零电流告警的逆变器")
            return results
        except Exception as e:
            print(f"查询零电流告警逆变器时出错: {str(e)}\nSQL语句:\n{query}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def _build_full_inverter_info(self, inverter_sn_list):
        """
        根据逆变器SN列表查询完整的逆变器信息
        """
        if not inverter_sn_list:
            return []
        
        placeholders = ','.join(['%s'] * len(inverter_sn_list))
        query = f"""
        SELECT 
            p.inverter_sn, 
            p.project_id, 
            p.plant_id, 
            sp.plant_name, 
            sp.tenant_id, 
            st.tenant_name,
            sp.grid_connection_date 
        FROM sys_inverter_project p 
        JOIN sys_plant sp 
          ON p.plant_id = sp.plant_id 
        JOIN sys_tenant st 
          ON sp.tenant_id = st.id 
         AND st.is_del = '0'        
        WHERE p.is_del = '0' 
          AND p.inverter_sn IN ({placeholders})
        ORDER BY st.tenant_name, sp.plant_name, p.inverter_sn
        """

        conn = None
        cursor = None
        try:
            conn = self._get_mysql_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, tuple(inverter_sn_list))
            results = cursor.fetchall()
            print(f"查询到 {len(results)} 个逆变器的完整信息")
            return results
        except Exception as e:
            print(f"查询逆变器完整信息时出错: {str(e)}\nSQL语句:\n{query}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def query_inverters_by_sn(self, inverter_sn):
        """
        按逆变器SN查询
        """
        query = """
        SELECT inverter_sn FROM `sys_inverter_project` 
        WHERE is_del='0' AND inverter_sn=%s
        """

        conn = None
        cursor = None
        try:
            conn = self._get_mysql_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, (inverter_sn,))
            results = cursor.fetchall()
            if not results:
                print(f"未找到逆变器 SN: {inverter_sn}")
                return []
            sn_list = [r['inverter_sn'] for r in results]
            return self._build_full_inverter_info(sn_list)
        except Exception as e:
            print(f"按SN查询逆变器时出错: {str(e)}\nSQL语句:\n{query}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def query_inverters_by_plant(self, plant_name):
        """
        按电站名称查询所有逆变器
        """
        query = """
        SELECT DISTINCT ip.inverter_sn 
        FROM sys_inverter_project ip 
        JOIN sys_plant p ON ip.plant_id = p.plant_id 
        WHERE ip.is_del = 0 
          AND p.plant_name = %s
        """

        conn = None
        cursor = None
        try:
            conn = self._get_mysql_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, (plant_name,))
            results = cursor.fetchall()
            if not results:
                print(f"未找到电站: {plant_name}")
                return []
            sn_list = [r['inverter_sn'] for r in results]
            print(f"电站 {plant_name} 下共有 {len(sn_list)} 个逆变器")
            return self._build_full_inverter_info(sn_list)
        except Exception as e:
            print(f"按电站查询逆变器时出错: {str(e)}\nSQL语句:\n{query}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def query_inverters_by_tenant(self, tenant_name):
        """
        按租户名称查询所有逆变器
        """
        query = """
        SELECT DISTINCT ip.inverter_sn 
        FROM sys_inverter_project ip 
        JOIN sys_plant p ON ip.plant_id = p.plant_id 
        JOIN sys_tenant t ON p.tenant_id = t.id 
        WHERE ip.is_del = 0 
          AND t.tenant_name = %s
        """

        conn = None
        cursor = None
        try:
            conn = self._get_mysql_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, (tenant_name,))
            results = cursor.fetchall()
            if not results:
                print(f"未找到租户: {tenant_name}")
                return []
            sn_list = [r['inverter_sn'] for r in results]
            print(f"租户 {tenant_name} 下共有 {len(sn_list)} 个逆变器")
            return self._build_full_inverter_info(sn_list)
        except Exception as e:
            print(f"按租户查询逆变器时出错: {str(e)}\nSQL语句:\n{query}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def query_string_config_by_inverter_sn(self, inverter_sn):
        """
        根据逆变器SN查询原组串接入配置
        """
        query = """
        SELECT * FROM sys_inverter_string_config 
        WHERE project_id IN (SELECT project_id FROM sys_inverter_project WHERE is_del='0' AND inverter_sn = %s)
        """

        conn = None
        cursor = None
        try:
            conn = self._get_mysql_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, (inverter_sn,))
            results = cursor.fetchall()
            print(f"逆变器 {inverter_sn} 查询到 {len(results)} 条组串配置记录")
            return results
        except Exception as e:
            print(f"查询组串配置时出错: {str(e)}\nSQL语句:\n{query}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def get_inverter_rated_power(self, inverter_sn):
        """
        获取逆变器额定功率
        """
        query = "SELECT rated_output_power FROM `sys_inverter_project` WHERE is_del='0' AND inverter_sn=%s"

        conn = None
        cursor = None
        try:
            conn = self._get_mysql_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, (inverter_sn,))
            result = cursor.fetchone()

            if result:
                rated_power = result.get('rated_output_power')
                print(f"逆变器 {inverter_sn} 的额定功率: {rated_power}")
                return rated_power
            else:
                print(f"未找到逆变器 {inverter_sn} 的额定功率信息")
                return None
        except Exception as e:
            print(f"查询逆变器 {inverter_sn} 额定功率时出错: {str(e)}\nSQL语句:\n{query}")
            return None
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def query_support_string(self, inverter_sn):
        """
        查询数据库中的support_string值
        """
        conn = None
        cursor = None
        try:
            print(f"正在查询逆变器 {inverter_sn} 的support_string值...")
            conn = self._get_mysql_connection()
            cursor = conn.cursor(dictionary=True)

            query = """
            SELECT support_string 
            FROM iot_device_model 
            WHERE model_name IN (
                SELECT model FROM sys_inverter_project 
                WHERE is_del='0' AND inverter_sn = %s
            )
            """

            cursor.execute(query, (inverter_sn,))
            result = cursor.fetchone()

            if result:
                support_string = result['support_string']
                print(f"逆变器 {inverter_sn} 的support_string值为: {support_string}")
                return support_string
            else:
                msg = f"未找到逆变器 {inverter_sn} 的support_string值，停止脚本"
                print(msg)
                log_auto_config("SYSTEM", inverter_sn, msg)
                sys.exit(1)

        except Exception as e:
            print(f"查询数据库时发生异常: {str(e)}\nSQL语句:\n{query}")
            if "Can't connect to MySQL server" in str(e) or "2003" in str(e):
                raise ConnectionError(f"数据库连接失败: {str(e)}")
            return None
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def get_inverter_string_count(self, inverter_sn):
        """
        获取逆变器的实际组串数量
        """
        conn = None
        cursor = None
        try:
            print(f"正在查询逆变器 {inverter_sn} 的组串数量...")
            conn = self._get_mysql_connection()
            cursor = conn.cursor(dictionary=True)

            query = """
            SELECT string_count 
            FROM sys_inverter_project 
            WHERE inverter_sn = %s AND is_del = '0'
            """

            cursor.execute(query, (inverter_sn,))
            result = cursor.fetchone()

            if result and result.get('string_count'):
                string_count = result['string_count']
                print(f"逆变器 {inverter_sn} 的组串数量为: {string_count}")
                return string_count
            else:
                print(f"未找到逆变器 {inverter_sn} 的组串数量，停止脚本")
                log_auto_config("SYSTEM", inverter_sn, "未找到逆变器组串数量，停止脚本")
                sys.exit(1)

        except Exception as e:
            print(f"查询逆变器组串数量时发生异常: {str(e)}，停止脚本\nSQL语句:\n{query}")
            log_auto_config("SYSTEM", inverter_sn, f"查询逆变器组串数量时发生异常: {str(e)}，停止脚本")
            sys.exit(1)
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def get_max_power_date(self, inverter_sn, start_date, end_date, power_threshold):
        """
        查询指定日期范围内发电量最高的日期
        """
        query = """
        SELECT date, product_power 
        FROM iot_device_data_day 
        WHERE device_no = %s 
          AND date BETWEEN %s AND %s 
          AND product_power > %s 
        ORDER BY product_power DESC 
        LIMIT 1
        """

        conn = None
        cursor = None
        try:
            conn = self._get_mysql_connection()
            cursor = conn.cursor(dictionary=True)
            if DEBUG_MODE:
                print(f"  [DEBUG] SQL查询: device_no={inverter_sn}, date BETWEEN {start_date} AND {end_date}, product_power > {power_threshold}")
                print(f"  [DEBUG] SQL语句: SELECT date, product_power FROM iot_device_data_day WHERE device_no = '{inverter_sn}' AND date BETWEEN '{start_date}' AND '{end_date}' AND product_power > {power_threshold} ORDER BY product_power DESC LIMIT 1")
            cursor.execute(query, (inverter_sn, start_date, end_date, power_threshold))
            result = cursor.fetchone()
            if DEBUG_MODE:
                print(f"  [DEBUG] SQL结果: {result}")
            return result
        except Exception as e:
            print(f"查询最高发电量日期时出错: {str(e)}\nSQL语句:\n{query}")
            return None
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def get_high_current_strings_by_date_range(self, inverter_sn, date_str, power_threshold, current_threshold, db_mode="main_only"):
        """
        根据日期查询当天有功功率>阈值且最大的一条数据的高电流组串
        db_mode: 
          - "main_only": 只查主库(MONGODB_CONFIG)
          - "main_then_cool": 优先查主库，查不到再查冷库(MONGODB_COOL_CONFIG)
          - "cool_only": 只查冷库
        返回: {'strings': [...], 'count': N, 'date': '...', 'used_cool': True/False} 或 None
        """
        clients = []
        try:
            if isinstance(date_str, str):
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            elif isinstance(date_str, datetime):
                date_obj = date_str
            else:
                date_obj = datetime.combine(date_str, datetime.min.time())
            date_str_fmt = date_obj.strftime('%Y-%m-%d')
            start_ts = int(date_obj.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
            end_ts = int(date_obj.replace(hour=23, minute=59, second=59, microsecond=999999).timestamp() * 1000)

            doc = None
            used_cool = False

            pipeline = [
                {"$match": {
                    "deviceno": inverter_sn,
                    "datats": {"$gte": start_ts, "$lte": end_ts}
                }},
                {"$addFields": {
                    "activePowerNum": {"$toDouble": "$dataobj.activePower"}
                }},
                {"$match": {
                    "activePowerNum": {"$gt": power_threshold}
                }},
                {"$sort": {"activePowerNum": -1}},
                {"$limit": 1}
            ]

            if DEBUG_MODE:
                print(f"  [DEBUG] MongoDB查询: deviceno={inverter_sn}, date={date_str_fmt}, datats范围={start_ts}~{end_ts}, power_threshold={power_threshold}")
                print(f"  [DEBUG] MongoDB pipeline:\n{json.dumps(pipeline, indent=2, default=str)}")

            def query_db(config, label):
                connection_string = get_mongodb_connection_string(config)
                client = MongoClient(connection_string, serverSelectionTimeoutMS=10000)
                clients.append(client)
                db = client[config["database"]]
                collection = db.iot_device_data
                if DEBUG_MODE:
                    print(f"  [DEBUG] 查询{label}: {config['host']}:{config['port']}")
                results = list(collection.aggregate(pipeline))
                if DEBUG_MODE:
                    print(f"  [DEBUG] {label}查询结果: {len(results)} 条")
                return results

            if db_mode == "main_only":
                results = query_db(MONGODB_CONFIG, "主库")
                if results:
                    doc = results[0]

            elif db_mode == "cool_only":
                results = query_db(MONGODB_COOL_CONFIG, "冷库")
                if results:
                    doc = results[0]
                    used_cool = True

            elif db_mode == "main_then_cool":
                results = query_db(MONGODB_CONFIG, "主库")
                if results:
                    doc = results[0]
                else:
                    print(f"主库未查询到逆变器 {inverter_sn} 在 {date_str_fmt} 的数据，尝试冷库...")
                    results = query_db(MONGODB_COOL_CONFIG, "冷库")
                    if results:
                        doc = results[0]
                        used_cool = True

            if not doc:
                return None

            dataobj = doc.get('dataobj', {})
            active_power = dataobj.get('activePower', 'N/A')
            if DEBUG_MODE:
                print(f"  [DEBUG] 选中文档: activePower={active_power}, datats={doc.get('datats')}")

            high_current_strings = set()
            for i in range(1, 31):
                current_key = f"stringDataCurrent{i}"
                current_value = dataobj.get(current_key)

                if current_value is not None:
                    try:
                        current_float = float(current_value)
                        if current_float > current_threshold:
                            high_current_strings.add(i)
                    except ValueError:
                        continue

            result = sorted(list(high_current_strings))
            return {'strings': result, 'count': len(result), 'date': date_str_fmt, 'used_cool': used_cool}

        except Exception as e:
            print(f"查询MongoDB时间范围数据时出错: {str(e)}")
            return None
        finally:
            for client in clients:
                try:
                    client.close()
                except:
                    pass


    def query_high_current_strings_by_inverter_newest(self, inverter_sn):
        """
        查询最新数据：获取当月发电量>额定功率的日期最新的一天的高电流组串
        """
        current_threshold = CURRENT_THRESHOLD
        power_ratio = POWER_RATIO
        generation_power_ratio = GENERATION_POWER_RATIO

        rated_power = self.get_inverter_rated_power(inverter_sn)

        if rated_power is not None and float(rated_power) > 0:
            rated_power = float(rated_power)
            active_power_threshold = rated_power * power_ratio
            generation_threshold = rated_power * generation_power_ratio
            print(f"逆变器 {inverter_sn} 额定功率: {rated_power}, activePower阈值(额定功率的{int(power_ratio * 100)}%): {active_power_threshold}, 发电量阈值(额定功率的{int(generation_power_ratio * 100)}%): {generation_threshold}")
        else:
            print(f"无法获取逆变器 {inverter_sn} 的额定功率，跳过")
            return {'strings': [], 'count': 0}

        now = datetime.now()
        current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        current_month_end = now.replace(hour=23, minute=59, second=59, microsecond=999999)

        print(f"正在查询逆变器 {inverter_sn} 当月({current_month_start.strftime('%Y-%m')})发电量>{generation_threshold}的最新日期...")
        
        query = """
        SELECT date, product_power 
        FROM iot_device_data_day 
        WHERE device_no = %s 
          AND date BETWEEN %s AND %s 
          AND product_power > %s 
        ORDER BY date DESC 
        LIMIT 1
        """

        conn = None
        cursor = None
        try:
            conn = self._get_mysql_connection()
            cursor = conn.cursor(dictionary=True)
            start_str = current_month_start.strftime('%Y-%m-%d')
            end_str = current_month_end.strftime('%Y-%m-%d')
            if DEBUG_MODE:
                print(f"  [DEBUG] SQL查询(最新): device_no={inverter_sn}, date BETWEEN {start_str} AND {end_str}, product_power > {generation_threshold}")
                print(f"  [DEBUG] SQL语句(最新): SELECT date, product_power FROM iot_device_data_day WHERE device_no = '{inverter_sn}' AND date BETWEEN '{start_str}' AND '{end_str}' AND product_power > {generation_threshold} ORDER BY date DESC LIMIT 1")
            cursor.execute(query, (inverter_sn, start_str, end_str, generation_threshold))
            result = cursor.fetchone()
            if DEBUG_MODE:
                print(f"  [DEBUG] SQL结果(最新): {result}")
        except Exception as e:
            print(f"查询最新发电量日期时出错: {str(e)}\nSQL语句:\n{query}")
            return {'strings': [], 'count': 0}
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

        if not result:
            print(f"逆变器 {inverter_sn} 当月未查询到发电量>{generation_threshold}的日期，返回空结果")
            return {'strings': [], 'count': 0}

        newest_date = result.get('date')
        print(f"逆变器 {inverter_sn} 当月发电量>{generation_threshold}的最新日期: {newest_date}，发电量: {result.get('product_power')}")

        print(f"正在查询逆变器 {inverter_sn} 在 {newest_date} 的高电流组串 (>{current_threshold}A)...")
        
        strings_result = self.get_high_current_strings_by_date_range(
            inverter_sn,
            newest_date,
            active_power_threshold,
            current_threshold,
            db_mode="main_only"
        )

        if not strings_result:
            print(f"逆变器 {inverter_sn} 在 {newest_date} 未查询到高电流组串数据")
            return {'strings': [], 'count': 0}

        print(f"最新数据查询到 {strings_result['count']} 个高电流组串: {strings_result['strings']}")
        return {'strings': strings_result['strings'], 'count': strings_result['count']}

    def _ask_user_choice(self, inverter_sn, newest_strings, oldest_strings, original_strings, total_strings, plant_name="", tenant_name=""):
        """
        当MANUAL_CONFIRM=True且新/旧数据有一个为空时，让用户选择处理方式
        返回: (high_current_strings, data_source, changed, update_reason)
        """
        newest_empty = len(newest_strings) == 0
        oldest_empty = len(oldest_strings) == 0

        print(f"\n{'='*60}")
        print(f"逆变器 {inverter_sn} 数据不完整，需要手动确认")
        print(f"  租户名称: {tenant_name}")
        print(f"  电站名称: {plant_name}")
        print(f"  最新数据: {newest_strings if newest_strings else '空'}")
        print(f"  最旧数据: {oldest_strings if oldest_strings else '空'}")
        print(f"  原配置: {original_strings if original_strings else '空'}")
        print(f"{'='*60}")
        print(f"请选择处理方式:")
        if not newest_empty:
            print(f"  1. 使用最新数据 ({newest_strings}, 数量: {len(newest_strings)})")
        else:
            print(f"  1. 使用最新数据 (空，不可用)")
        if not oldest_empty:
            print(f"  2. 使用最旧数据 ({oldest_strings}, 数量: {len(oldest_strings)})")
        else:
            print(f"  2. 使用最旧数据 (空，不可用)")
        print(f"  3. 保持原配置 ({original_strings if original_strings else '空'}, 数量: {len(original_strings)})")
        print(f"  4. 跳过该逆变器")
        print()

        while True:
            choice = input("请输入选项 (1-4): ").strip()
            if choice == '1':
                if newest_empty:
                    print("最新数据为空，不可用，请重新选择")
                    continue
                print(f"已选择使用最新数据: {newest_strings}")
                return newest_strings, "newest", True, "手动选择使用最新数据"
            elif choice == '2':
                if oldest_empty:
                    print("最旧数据为空，不可用，请重新选择")
                    continue
                print(f"已选择使用最旧数据: {oldest_strings}")
                return oldest_strings, "oldest", True, "手动选择使用最旧数据"
            elif choice == '3':
                print(f"已选择保持原配置: {original_strings}")
                return original_strings, "original", False, "手动选择保持原配置"
            elif choice == '4':
                print(f"已选择跳过该逆变器")
                return [], "skip", False, "手动选择跳过"
            else:
                print("无效选项，请输入 1-4")

    def _generate_months_list(self, grid_month_start, count, current_month_start):
        """
        生成需要查询的月份列表（由近到远排序）
        """
        previous_month_end = current_month_start - timedelta(microseconds=1)
        previous_month_start = previous_month_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # 并网时间为当月则跳过
        if grid_month_start >= current_month_start:
            return []

        # 并网时间为上一个月则只查上一个月
        if grid_month_start == previous_month_start:
            return self._get_month_range(previous_month_start)

        # 生成从并网月份到上一个月的所有月份（由远到近）
        all_months = []
        cursor = grid_month_start
        while cursor <= previous_month_start:
            all_months.append(cursor)
            if cursor.month == 12:
                cursor = datetime(cursor.year + 1, 1, 1)
            else:
                cursor = datetime(cursor.year, cursor.month + 1, 1)

        # 并网当月和次月
        grid_next = None
        if grid_month_start.month == 12:
            grid_next = datetime(grid_month_start.year + 1, 1, 1)
        else:
            grid_next = datetime(grid_month_start.year, grid_month_start.month + 1, 1)

        query_months = []
        # 先加入次月（如果在范围内）
        if grid_next <= previous_month_start:
            query_months.append(grid_next)
        # 再加入并网当月
        query_months.append(grid_month_start)

        if count == "all":
            # 所有月份
            for m in all_months:
                if m not in query_months:
                    query_months.append(m)
        elif isinstance(count, int) and count > 0:
            # 最近N个月（不包括并网当月和次月）
            recent_months = []
            cursor = previous_month_start
            for _ in range(count):
                if cursor >= grid_next:
                    if cursor not in query_months:
                        recent_months.append(cursor)
                if cursor.month == 1:
                    cursor = datetime(cursor.year - 1, 12, 1)
                else:
                    cursor = datetime(cursor.year, cursor.month - 1, 1)
            query_months.extend(recent_months)
        elif isinstance(count, int) and count == 0:
            # 只查并网当月和次月，已经加入了
            pass

        # 由近到远排序
        query_months.sort(reverse=True)

        # 转换为(month_start, month_end)元组
        result = []
        for m in query_months:
            ranges = self._get_month_range(m)
            result.extend(ranges)
        return result

    def _get_month_range(self, month_start):
        """获取指定月份的起止时间"""
        if month_start.month == 12:
            month_end = datetime(month_start.year + 1, 1, 1) - timedelta(microseconds=1)
        else:
            month_end = datetime(month_start.year, month_start.month + 1, 1) - timedelta(microseconds=1)
        return [(month_start, month_end)]

    def _determine_initial_db_mode(self, grid_month_start, current_month_start):
        """根据并网时间与当前时间的月份差决定初始MongoDB查询模式"""
        months_diff = (current_month_start.year - grid_month_start.year) * 12 + (current_month_start.month - grid_month_start.month)
        if months_diff <= 5:
            return "main_only"
        else:
            return "main_then_cool"

    def query_high_current_strings_by_inverter_oldest(self, inverter_sn, grid_connection_date=None):
        """
        查询最旧数据：按RECENT_MONTHS_COUNT配置查询指定月份范围的高电流组串
        由近到远查询， MongoDB策略：5个月内只用主库，超过5个月主库优先→冷库回退
        """
        current_threshold = CURRENT_THRESHOLD
        power_ratio = POWER_RATIO
        generation_power_ratio = GENERATION_POWER_RATIO

        rated_power = self.get_inverter_rated_power(inverter_sn)

        if rated_power is not None and float(rated_power) > 0:
            rated_power = float(rated_power)
            active_power_threshold = rated_power * power_ratio
            generation_threshold = rated_power * generation_power_ratio
            print(f"逆变器 {inverter_sn} 额定功率: {rated_power}, activePower阈值(额定功率的{int(power_ratio * 100)}%): {active_power_threshold}, 发电量阈值(额定功率的{int(generation_power_ratio * 100)}%): {generation_threshold}")
        else:
            print(f"无法获取逆变器 {inverter_sn} 的额定功率，跳过")
            return {'strings': [], 'count': 0}

        if not grid_connection_date:
            print(f"逆变器 {inverter_sn} 未提供并网时间，返回空结果")
            return {'strings': [], 'count': 0}

        try:
            date_str = str(grid_connection_date)
            if ' ' in date_str:
                grid_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            else:
                grid_date = datetime.strptime(date_str, '%Y-%m-%d')
        except:
            print(f"逆变器 {inverter_sn} 并网时间格式错误: {grid_connection_date}")
            return {'strings': [], 'count': 0}

        current_month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        grid_month_start = grid_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # 并网时间为当月则跳过
        if grid_month_start >= current_month_start:
            print(f"逆变器 {inverter_sn} 并网时间为当月，跳过旧数据查询")
            return {'strings': [], 'count': 0}

        # 生成需要查询的月份列表
        months_to_query = self._generate_months_list(grid_month_start, RECENT_MONTHS_COUNT, current_month_start)

        if not months_to_query:
            print(f"逆变器 {inverter_sn} 没有需要查询的月份，跳过旧数据查询")
            return {'strings': [], 'count': 0}

        # 决定初始MongoDB查询模式
        db_mode = self._determine_initial_db_mode(grid_month_start, current_month_start)
        print(f"逆变器 {inverter_sn} MongoDB查询模式: {db_mode}")

        # 由近到远查询各月份的发电量最高日期
        query_dates = []
        consecutive_empty = 0
        for month_start, month_end in months_to_query:
            result = self.get_max_power_date(
                inverter_sn,
                month_start.strftime('%Y-%m-%d'),
                month_end.strftime('%Y-%m-%d'),
                generation_threshold
            )
            if result:
                query_dates.append((result.get('date'), month_start))
                print(f"逆变器 {inverter_sn} {month_start.strftime('%Y-%m')}发电量最高日期: {result.get('date')}")
                consecutive_empty = 0
            else:
                print(f"逆变器 {inverter_sn} {month_start.strftime('%Y-%m')}未查询到发电量>{generation_threshold}的日期")
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    print(f"逆变器 {inverter_sn} 连续3个月未查询到数据，停止继续查询")
                    break

        if not query_dates:
            print(f"逆变器 {inverter_sn} 所有月份都未查询到发电量>{generation_threshold}的日期，跳过该逆变器")
            return {'strings': [], 'count': 0}

        print(f"共查询到 {len(query_dates)} 个有效日期，开始查询高电流组串...")

        # 由近到远查询MongoDB
        all_results = []
        cool_confirmed = False  # 冷库是否已确认有数据
        consecutive_empty_mongo = 0

        for date_val, month_start in query_dates:
            # 决定当前查询的db_mode
            if cool_confirmed:
                current_db_mode = "cool_only"
            elif db_mode == "main_only":
                current_db_mode = "main_only"
            else:
                current_db_mode = "main_then_cool"

            strings_result = self.get_high_current_strings_by_date_range(
                inverter_sn,
                date_val,
                active_power_threshold,
                current_threshold,
                db_mode=current_db_mode
            )

            if strings_result:
                all_results.append(strings_result)
                print(f"  {date_val}: {strings_result['count']}个高电流组串 (库: {'冷库' if strings_result.get('used_cool') else '主库'})")
                if strings_result.get('used_cool') and not cool_confirmed:
                    cool_confirmed = True
                    print(f"逆变器 {inverter_sn} 冷库查询成功，后续所有查询使用冷库")
                consecutive_empty_mongo = 0
            else:
                print(f"  {date_val}: 无数据，跳过")
                consecutive_empty_mongo += 1
                if consecutive_empty_mongo >= 3:
                    print(f"逆变器 {inverter_sn} 连续3个月MongoDB查询无数据，停止继续查询")
                    break

        if not all_results:
            print(f"逆变器 {inverter_sn} 所有日期都未查询到高电流组串数据，跳过该逆变器")
            return {'strings': [], 'count': 0}

        # 取高电流组串数量最多且时间最近的做为旧数据
        max_count = max(r['count'] for r in all_results)
        best_results = [r for r in all_results if r['count'] == max_count]
        best_result = best_results[0]

        print(f"最旧数据查询到 {best_result['count']} 个高电流组串: {best_result['strings']} (来自{best_result['date']})")
        return {'strings': best_result['strings'], 'count': best_result['count']}

    def calculate_string_status(self, inverter_sn, plant_name, total_strings, original_strings, tenant_name="", grid_connection_date=None):
        """
        根据最新/最旧数据计算组串接入状态
        返回: (new_status_list, data_source, changed, update_reason, newest_strings, oldest_strings)
        """
        newest_result = self.query_high_current_strings_by_inverter_newest(inverter_sn)
        oldest_result = self.query_high_current_strings_by_inverter_oldest(inverter_sn, grid_connection_date)

        newest_strings = newest_result.get('strings', [])
        oldest_strings = oldest_result.get('strings', [])
        newest_count = newest_result.get('count', 0)
        oldest_count = oldest_result.get('count', 0)

        original_count = len(original_strings)

        filtered_newest_strings = [s for s in newest_strings if s <= total_strings]
        filtered_oldest_strings = [s for s in oldest_strings if s <= total_strings]

        if len(filtered_newest_strings) < len(newest_strings):
            removed_strings = [s for s in newest_strings if s > total_strings]
            print(f"  警告: 最新数据中发现超出组串范围({total_strings}个)的组串: {removed_strings}，已过滤")
        if len(filtered_oldest_strings) < len(oldest_strings):
            removed_strings = [s for s in oldest_strings if s > total_strings]
            print(f"  警告: 最旧数据中发现超出组串范围({total_strings}个)的组串: {removed_strings}，已过滤")

        newest_strings = filtered_newest_strings
        oldest_strings = filtered_oldest_strings
        newest_count = len(newest_strings)
        oldest_count = len(oldest_strings)

        print(f"  逆变器 {inverter_sn} 高电流组串对比:")
        print(f"    原组串接入配置: {original_strings}, 数量: {original_count}")
        print(f"    最新数据高电流组串: {newest_strings}, 数量: {newest_count}")
        print(f"    最旧数据高电流组串: {oldest_strings}, 数量: {oldest_count}")

        original_set = set(original_strings)
        newest_set = set(newest_strings)
        oldest_set = set(oldest_strings)

        newest_oldest_same = newest_count == oldest_count and set(newest_strings) == set(oldest_strings)
        both_empty = newest_count == 0 and oldest_count == 0

        high_current_strings = []
        data_source = ""
        changed = False
        update_reason = ""

        if original_count > 0:
            mod_record = read_mod_record()

            if inverter_sn in mod_record:
                mod_count = mod_record[inverter_sn]
                print(f"  结论: 该逆变器在整改记录表中，整改后组串数量: {mod_count}")
                if newest_count < mod_count:
                    # 1a-1: 最新数据<整改数，使用最新数据+从原配置补齐到整改数
                    print(f"  最新数据高电流组串数量({newest_count})<整改后组串数量({mod_count})，补齐原配置到整改数 (1a)")
                    selected = list(newest_strings)
                    remaining = [s for s in original_strings if s not in newest_set][:mod_count - newest_count]
                    selected.extend(remaining)
                    high_current_strings = selected
                    data_source = "newest_padded"
                    changed = True
                    update_reason = f"最新数据({newest_count})<整改数({mod_count})，使用最新数据+原配置补齐至整改数"
                elif newest_count == mod_count:
                    # 1a-2: 最新数据=整改数，直接使用最新数据
                    print(f"  最新数据高电流组串数量({newest_count})=整改后组串数量({mod_count})，使用最新数据 (1a)")
                    high_current_strings = newest_strings
                    data_source = "newest"
                    changed = True
                    update_reason = f"最新数据({newest_count})=整改数({mod_count})，使用最新数据配置"
                else:
                    # 1a-3: 最新数据>整改数，使用最新数据+删除SN记录
                    print(f"  最新数据高电流组串数量({newest_count})>整改后组串数量({mod_count})，使用最新数据并删除该SN记录 (1a)")
                    log_mod_record(plant_name, inverter_sn, mod_count, newest_count, "最新数据高电流组串数量多于整改后组串数量，使用最新数据配置并删除整改记录")
                    delete_mod_record(inverter_sn)
                    high_current_strings = newest_strings
                    data_source = "newest"
                    changed = True
                    update_reason = f"最新数据({newest_count})>整改数({mod_count})，使用最新数据配置并删除整改记录"

            elif both_empty:
                print(f"  结论: 原组串接入配置不为空，但最新/最旧数据都为空，需要检查support_string (1d)")
                support_string = self.query_support_string(inverter_sn)
                if support_string == 0:
                    if original_count == total_strings:
                        print(f"  support_string=0，但原配置已全接入({original_count}/{total_strings})，保持原配置无需更新 (1d)")
                        high_current_strings = original_strings
                        data_source = "original"
                    else:
                        print(f"  support_string=0（不支持组串监测），将所有组串设置为接入 (1d)")
                        high_current_strings = list(range(1, total_strings + 1))
                        data_source = "all_strings"
                        changed = True
                        update_reason = "最新和最旧数据都为空，不支持组串监测，将所有组串设置为接入"
                else:
                    print(f"  support_string={support_string}（支持组串监测），保持原配置 (1d)")
                    high_current_strings = original_strings
                    data_source = "original"

            elif newest_count == 0 and oldest_count > 0:
                if MANUAL_CONFIRM:
                    print(f"  最新数据为空，最旧数据不为空，触发手动确认 (1e/1f)")
                    result_strings, result_source, result_changed, result_reason = self._ask_user_choice(
                        inverter_sn, newest_strings, oldest_strings, original_strings, total_strings, plant_name, tenant_name
                    )
                    high_current_strings = result_strings
                    data_source = result_source
                    changed = result_changed
                    update_reason = result_reason
                elif oldest_count == original_count:
                    print(f"  最新数据为空，最旧数据不为空且数量({oldest_count})与原配置({original_count})一致，保持原配置 (1e)")
                    high_current_strings = original_strings
                    data_source = "original"
                else:
                    print(f"  最新数据为空，最旧数据不为空且数量({oldest_count})与原配置({original_count})不一致，使用高电流组串数量最多的进行配置 (1f)")
                    if original_count >= oldest_count:
                        print(f"  原配置({original_count}) >= 最旧数据({oldest_count})，保持原配置 (1f)")
                        high_current_strings = original_strings
                        data_source = "original"
                    else:
                        print(f"  最旧数据({oldest_count}) > 原配置({original_count})，使用最旧数据配置 (1f)")
                        high_current_strings = oldest_strings
                        data_source = "oldest"
                        changed = True
                        update_reason = "最新数据为空，使用最旧数据配置"

            elif oldest_count == 0 and newest_count > 0:
                if MANUAL_CONFIRM:
                    print(f"  最旧数据为空，最新数据不为空，触发手动确认 (1g/1h)")
                    result_strings, result_source, result_changed, result_reason = self._ask_user_choice(
                        inverter_sn, newest_strings, oldest_strings, original_strings, total_strings, plant_name, tenant_name
                    )
                    high_current_strings = result_strings
                    data_source = result_source
                    changed = result_changed
                    update_reason = result_reason
                elif newest_count == original_count:
                    if newest_set == original_set:
                        print(f"  最旧数据为空，最新数据不为空且数量({newest_count})与原配置({original_count})一致，顺序也一致，保持原配置 (1g)")
                        high_current_strings = original_strings
                        data_source = "original"
                    else:
                        print(f"  最旧数据为空，最新数据不为空且数量({newest_count})与原配置({original_count})一致，但顺序不一致，使用最新数据配置 (1g)")
                        high_current_strings = newest_strings
                        data_source = "newest"
                        changed = True
                        update_reason = "原组串接入配置不为空，最新数据数量与原配置一致但顺序不一致"
                else:
                    print(f"  最旧数据为空，最新数据不为空且数量({newest_count})与原配置({original_count})不一致，使用高电流组串数量最多的进行配置 (1h)")
                    if original_count >= newest_count:
                        print(f"  原配置({original_count}) >= 最新数据({newest_count})，保持原配置 (1h)")
                        high_current_strings = original_strings
                        data_source = "original"
                    else:
                        print(f"  最新数据({newest_count}) > 原配置({original_count})，使用最新数据配置 (1h)")
                        high_current_strings = newest_strings
                        data_source = "newest"
                        changed = True
                        update_reason = "最旧数据为空，使用最新数据配置"

            elif newest_count == original_count and oldest_count == original_count:
                if newest_set == original_set:
                    print(f"  原配置、最新、最旧三者数量且顺序一致，保持原配置 (1c)")
                    high_current_strings = original_strings
                    data_source = "original"
                else:
                    print(f"  原配置、最新、最旧三者数量一致但顺序不一致，使用最新数据配置 (1b)")
                    high_current_strings = newest_strings
                    data_source = "newest"
                    changed = True
                    update_reason = "原组串接入配置不为空，且与最新/最旧数据高电流数量一致但顺序不一致"

            elif newest_count == original_count and newest_count > oldest_count:
                if newest_set == original_set:
                    print(f"  原配置与最新数量一致({original_count})且大于最旧({oldest_count})，顺序一致，保持原配置 (1j)")
                    high_current_strings = original_strings
                    data_source = "original"
                else:
                    print(f"  原配置与最新数量一致({original_count})且大于最旧({oldest_count})，顺序不一致，使用最新数据配置 (1j)")
                    high_current_strings = newest_strings
                    data_source = "newest"
                    changed = True
                    update_reason = "原组串接入配置不为空，原配置与最新数量一致大于最旧但顺序不一致"

            elif oldest_count == original_count and oldest_count > newest_count:
                print(f"  原配置与最旧数量一致({original_count})且大于最新({newest_count})，保持原配置 (1k)")
                high_current_strings = original_strings
                data_source = "original"

            else:
                print(f"  三者数量都不相等或其中一个大于另外两个，使用高电流组串数量最多的进行配置 (1i)")
                max_count = max(original_count, newest_count, oldest_count)
                if original_count == max_count:
                    print(f"  原配置({original_count})为最大，保持原配置 (1i)")
                    high_current_strings = original_strings
                    data_source = "original"
                elif newest_count == max_count:
                    print(f"  最新数据({newest_count})为最大，使用最新数据配置 (1i)")
                    high_current_strings = newest_strings
                    data_source = "newest"
                    changed = True
                    update_reason = "原组串接入配置不为空，使用高电流组串数量最多的最新数据配置"
                else:
                    print(f"  最旧数据({oldest_count})为最大，使用最旧数据配置 (1i)")
                    high_current_strings = oldest_strings
                    data_source = "oldest"
                    changed = True
                    update_reason = "原组串接入配置不为空，使用高电流组串数量最多的最旧数据配置"
        else:
            if both_empty:
                print(f"逆变器 {inverter_sn} 没有发现高电流组串")

                try:
                    support_string = self.query_support_string(inverter_sn)

                    if support_string == 0:
                        print(f"support_string=0（不支持组串监测），将所有组串设置为接入")
                        high_current_strings = list(range(1, total_strings + 1))
                        data_source = "all_strings"
                        changed = True
                        update_reason = "最新和最旧数据都为空"
                    else:
                        print(f"support_string={support_string}（支持组串监测），跳过")
                        high_current_strings = []
                        data_source = "skip"
                except ConnectionError as e:
                    print(f"数据库连接失败: {str(e)}")
                    raise e
            elif newest_oldest_same:
                print(f"  结论: 原组串接入配置为空，最新和最旧数据一致，使用最新数据配置")
                high_current_strings = newest_strings
                data_source = "newest"
                changed = True
                update_reason = "最新和最旧数据不为空且一致"
            else:
                one_empty = (newest_count == 0) != (oldest_count == 0)

                if one_empty:
                    if MANUAL_CONFIRM:
                        result_strings, result_source, result_changed, result_reason = self._ask_user_choice(
                            inverter_sn, newest_strings, oldest_strings, original_strings, total_strings, plant_name, tenant_name
                        )
                        high_current_strings = result_strings
                        data_source = result_source
                        changed = result_changed
                        update_reason = result_reason
                    elif newest_count > 0:
                        print(f"  最新数据有高电流组串({newest_count}个)，最旧数据为空，使用最新数据配置")
                        high_current_strings = newest_strings
                        data_source = "newest"
                        changed = True
                        update_reason = "最新/最旧数据有一个为空一个不为空"
                    else:
                        print(f"  最旧数据有高电流组串({oldest_count}个)，最新数据为空，使用最旧数据配置")
                        high_current_strings = oldest_strings
                        data_source = "oldest"
                        changed = True
                        update_reason = "最新/最旧数据有一个为空一个不为空"
                else:
                    print(f"  结论: 原组串接入配置为空，最新和最旧数据不为空但不一致，使用高电流组串数量最多的数据进行配置")
                    if newest_count >= oldest_count:
                        high_current_strings = newest_strings
                        data_source = "newest"
                        print(f"  选择最新数据({newest_count}个)进行配置")
                    else:
                        high_current_strings = oldest_strings
                        data_source = "oldest"
                        print(f"  选择最旧数据({oldest_count}个)进行配置")
                    changed = True
                    update_reason = "最新和最旧数据不为空但不一致"

        new_status_list = []
        for idx in range(total_strings):
            string_num = idx + 1
            if string_num in high_current_strings:
                new_status_list.append(1)
            else:
                new_status_list.append(2)

        return new_status_list, data_source, changed, update_reason, newest_strings, oldest_strings

    def update_string_config(self, project_id, string_no, status):
        """
        更新sys_inverter_string_config表
        """
        query = """
        UPDATE sys_inverter_string_config 
        SET status = %s, update_time = NOW() 
        WHERE project_id = %s AND string_no = %s
        """

        conn = None
        cursor = None
        try:
            conn = self._get_mysql_connection()
            cursor = conn.cursor()
            cursor.execute(query, (status, project_id, string_no))
            conn.commit()
            return True
        except Exception as e:
            print(f"更新组串配置时出错: {str(e)}\nSQL语句:\n{query}")
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def insert_string_config(self, plant_id, tenant_id, project_id, string_no, status):
        """
        插入新的组串配置记录
        """
        query = """
        INSERT INTO sys_inverter_string_config (plant_id, tenant_id, project_id, string_no, status, create_time, update_time)
        VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
        """

        conn = None
        cursor = None
        try:
            conn = self._get_mysql_connection()
            cursor = conn.cursor()
            cursor.execute(query, (plant_id, tenant_id, project_id, string_no, status))
            conn.commit()
            return True
        except Exception as e:
            print(f"插入组串配置时出错: {str(e)}\nSQL语句:\n{query}")
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def get_project_id_by_inverter_sn(self, inverter_sn):
        """
        根据逆变器SN获取project_id
        """
        query = "SELECT project_id FROM sys_inverter_project WHERE is_del='0' AND inverter_sn = %s"

        conn = None
        cursor = None
        try:
            conn = self._get_mysql_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, (inverter_sn,))
            result = cursor.fetchone()

            if result:
                return result.get('project_id')
            else:
                return None
        except Exception as e:
            print(f"查询project_id时出错: {str(e)}")
            return None
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def clear_inverter_cache(self, inverter_sn, project_id):
        """
        清除对应逆变器的组串配置缓存
        使用Redis删除:
        1. inverter_string_config_key:{inverter_sn}
        2. inverter:project:sn:{inverter_sn}
        3. inverter_string_config_projectId_key:{project_id}
        4. HASH类型: inverter:string:status:hash 中的 {project_id} 字段
        返回: (success, message)
        说明: HASH缓存字段不存在或已过期也算成功
        """
        if not inverter_sn:
            return False, "逆变器SN为空"
        
        if not project_id:
            return False, "项目ID为空"
        
        cache_keys = [
            f"inverter_string_config_key:{inverter_sn}",
            f"inverter:project:sn:{inverter_sn}",
            f"inverter_string_config_projectId_key:{project_id}"
        ]
        
        hash_cache_key = "inverter:string:status:hash"
        hash_field = str(project_id)
        
        print(f"正在清除逆变器 {inverter_sn} 的缓存，keys: {cache_keys}")
        print(f"正在清除HASH缓存: {hash_cache_key} 的字段: {hash_field}")
        
        r = None
        try:
            r = redis.Redis(
                host=REDIS_CONFIG["host"],
                port=REDIS_CONFIG["port"],
                password=REDIS_CONFIG["password"],
                db=REDIS_CONFIG["db"],
                decode_responses=True,
                socket_timeout=5
            )
            
            all_success = True
            for cache_key in cache_keys:
                try:
                    result = r.delete(cache_key)
                    if result > 0:
                        print(f"成功清除缓存: {cache_key}")
                    else:
                        print(f"缓存不存在或已过期: {cache_key}")
                except Exception as e:
                    print(f"清除缓存 {cache_key} 失败: {str(e)}")
                    all_success = False
            
            try:
                hash_result = r.hdel(hash_cache_key, hash_field)
                if hash_result > 0:
                    print(f"成功清除HASH缓存: {hash_cache_key} 字段: {hash_field}")
                else:
                    print(f"HASH缓存字段不存在或已过期: {hash_cache_key} 字段: {hash_field}")
            except Exception as e:
                print(f"清除HASH缓存失败: {str(e)}")
                all_success = False
            
            return all_success, "成功" if all_success else "部分失败"
        except redis.ConnectionError as e:
            return False, f"Redis连接失败: {str(e)}"
        except Exception as e:
            return False, f"清除缓存时出错: {str(e)}"
        finally:
            if r:
                try:
                    r.close()
                except:
                    pass

    def process_inverter(self, inverter_info):
        """
        处理单个逆变器的组串配置
        """
        inverter_sn = inverter_info.get('inverter_sn')
        plant_name = inverter_info.get('plant_name', '未知电站')
        tenant_name = inverter_info.get('tenant_name', '未知租户')
        project_id = inverter_info.get('project_id')
        plant_id = inverter_info.get('plant_id')
        tenant_id = inverter_info.get('tenant_id')
        grid_connection_date = inverter_info.get('grid_connection_date')

        print(f"\n{'='*60}")
        print(f"处理逆变器: {inverter_sn}")
        print(f"租户: {tenant_name}")
        print(f"电站: {plant_name}")
        print(f"项目ID: {project_id}")
        print(f"电站ID: {plant_id}")
        print(f"租户ID: {tenant_id}")
        print(f"并网时间: {grid_connection_date}")

        if not project_id:
            project_id = self.get_project_id_by_inverter_sn(inverter_sn)
            if not project_id:
                print(f"无法获取逆变器 {inverter_sn} 的project_id，跳过")
                return 'no_data'

        if not plant_id or not tenant_id:
            print(f"无法获取逆变器 {inverter_sn} 的plant_id或tenant_id，跳过")
            return 'no_data'

        original_config = self.query_string_config_by_inverter_sn(inverter_sn)

        original_strings = []
        total_strings = 0
        existing_string_nos = set()

        for config in original_config:
            string_no = config.get('string_no')
            status = config.get('status')
            if string_no and status == 1:
                original_strings.append(string_no)
            if string_no:
                existing_string_nos.add(string_no)
                total_strings = max(total_strings, string_no)

        if total_strings == 0:
            total_strings = self.get_inverter_string_count(inverter_sn)

        print(f"原组串接入配置: {original_strings}")
        print(f"组串总数: {total_strings}")

        try:
            new_status_list, data_source, changed, update_reason, newest_strings, oldest_strings = self.calculate_string_status(
                inverter_sn, plant_name, total_strings, original_strings, tenant_name, grid_connection_date
            )
        except ConnectionError as e:
            print(f"数据库连接失败，跳过逆变器 {inverter_sn}: {str(e)}")
            return 'no_data'

        if data_source == "skip":
            print(f"跳过逆变器 {inverter_sn}")
            return 'no_data'

        if not changed and data_source == "original":
            print(f"逆变器 {inverter_sn} 配置未变化，无需更新")
            return 'no_change'

        print(f"准备更新组串配置:")
        for idx, status in enumerate(new_status_list):
            string_no = idx + 1
            print(f"  组串 {string_no}: {'接入' if status == 1 else '未接入'}")

        update_success = True
        for idx, status in enumerate(new_status_list):
            string_no = idx + 1

            if string_no in existing_string_nos:
                success = self.update_string_config(project_id, string_no, status)
            else:
                success = self.insert_string_config(plant_id, tenant_id, project_id, string_no, status)

            if not success:
                update_success = False
                break

        update_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if update_success:
            print(f"逆变器 {inverter_sn} 组串配置更新成功")
            
            cache_success, cache_msg = self.clear_inverter_cache(inverter_sn, project_id)
            
            data_source_desc = {
                'newest': '最新数据',
                'oldest': '最旧数据',
                'original': '原配置',
                'all_strings': '全部组串'
            }.get(data_source, data_source)
            
            if changed:
                new_strings = [idx + 1 for idx, status in enumerate(new_status_list) if status == 1]
                log_entry = f"""修改时间：{update_time}
租户名称：{tenant_name}
电站名称：{plant_name}
逆变器SN:{inverter_sn}
更新原因：{update_reason}
组串配置更新结果：成功
逆变器缓存清除结果：{'成功' if cache_success else '失败'}
更新数据源：{data_source_desc}
{_pad_label('原组串接入配置:')}{original_strings}, 数量: {len(original_strings)}
{_pad_label('最新数据高电流组串:')}{newest_strings}, 数量: {len(newest_strings)}
{_pad_label('最旧数据高电流组串:')}{oldest_strings}, 数量: {len(oldest_strings)}

"""
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(log_entry)
            
            return 'success' if cache_success else 'no_data'
        else:
            print(f"逆变器 {inverter_sn} 组串配置更新失败")
            
            if changed:
                new_strings = [idx + 1 for idx, status in enumerate(new_status_list) if status == 1]
                log_entry = f"""修改时间：{update_time}
租户名称：{tenant_name}
电站名称：{plant_name}
逆变器SN:{inverter_sn}
更新原因：{update_reason}
组串配置更新结果：失败
逆变器缓存清除结果：未执行
更新数据源：未知
{_pad_label('原组串接入配置:')}{original_strings}, 数量: {len(original_strings)}
{_pad_label('最新数据高电流组串:')}{newest_strings}, 数量: {len(newest_strings)}
{_pad_label('最旧数据高电流组串:')}{oldest_strings}, 数量: {len(oldest_strings)}

"""
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(log_entry)
            
            return 'no_data'

    def _show_menu(self):
        """
        显示操作菜单并返回用户选择
        """
        print(f"\n{'='*80}")
        print(f"组串自动配置检查脚本")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        print(f"\n请选择执行场景:")
        print(f"  1. 只查询没有配置组串记录的逆变器")
        print(f"  2. 只查询零电流告警组串的逆变器")
        print(f"  3. 按逆变器SN查询并配置")
        print(f"  4. 按电站名称查询所有逆变器并配置")
        print(f"  5. 按租户名称查询所有逆变器并配置")
        print(f"  6. 查询1+2 (没有配置+零电流告警)")
        print(f"  0. 退出")
        print()
        
        while True:
            choice = input("请输入选项 (0-6): ").strip()
            if choice in ('0', '1', '2', '3', '4', '5', '6'):
                return choice
            print("无效选项，请输入 0-6")

    def _get_inverters_by_scenario(self, choice):
        """
        根据选项获取需要处理的逆变器列表
        """
        if choice == '1':
            return self.query_inverters_without_config()
        
        elif choice == '2':
            return self.query_inverters_with_zero_current_alarm()
        
        elif choice == '3':
            sn = input("请输入逆变器SN: ").strip()
            if not sn:
                print("逆变器SN不能为空")
                return []
            return self.query_inverters_by_sn(sn)
        
        elif choice == '4':
            plant_name = input("请输入电站名称: ").strip()
            if not plant_name:
                print("电站名称不能为空")
                return []
            return self.query_inverters_by_plant(plant_name)
        
        elif choice == '5':
            tenant_name = input("请输入租户名称: ").strip()
            if not tenant_name:
                print("租户名称不能为空")
                return []
            return self.query_inverters_by_tenant(tenant_name)
        
        elif choice == '6':
            inverters_without_config = self.query_inverters_without_config()
            inverters_with_alarm = self.query_inverters_with_zero_current_alarm()

            all_inverter_sns = set()
            all_inverters = []

            for inv in inverters_without_config:
                sn = inv.get('inverter_sn')
                if sn and sn not in all_inverter_sns:
                    all_inverter_sns.add(sn)
                    all_inverters.append(inv)

            for inv in inverters_with_alarm:
                sn = inv.get('inverter_sn')
                if sn and sn not in all_inverter_sns:
                    all_inverter_sns.add(sn)
                    all_inverters.append(inv)

            print(f"\n合并后共需处理 {len(all_inverters)} 个逆变器")
            return all_inverters
        
        return []

    def run(self):
        """
        主执行函数
        """
        choice = self._show_menu()
        
        if choice == '0':
            print("退出脚本")
            return

        all_inverters = self._get_inverters_by_scenario(choice)
        
        if not all_inverters:
            print("没有需要处理的逆变器，退出")
            return

        success_count = 0
        no_change_count = 0
        no_data_count = 0

        for inverter_info in all_inverters:
            try:
                result = self.process_inverter(inverter_info)
                if result == 'success':
                    success_count += 1
                elif result == 'no_change':
                    no_change_count += 1
                elif result == 'no_data':
                    no_data_count += 1
                else:
                    no_change_count += 1
            except Exception as e:
                print(f"处理逆变器 {inverter_info.get('inverter_sn')} 时发生异常: {str(e)}")
                no_data_count += 1

        total_count = len(all_inverters)
        print(f"\n{'='*80}")
        print(f"组串自动配置检查脚本执行完成")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总共 {total_count} 个逆变器需要处理，配置成功 {success_count} 个，无需配置 {no_change_count} 个，无数据导致无法配置 {no_data_count} 个")
        print(f"{'='*80}")

        log_auto_config("SYSTEM", "ALL", f"自动配置任务完成，成功: {success_count}, 无需配置: {no_change_count}, 无数据: {no_data_count}")


if __name__ == "__main__":
    config = CheckStringsAutoConfig()
    config.run()