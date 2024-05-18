import os
import sqlite3
import label_studio_sdk
import logging

# 假设这些是全局变量或者从配置文件中加载
LABEL_STUDIO_DATA_PATH = "E:\docker\label-studio\data"
LABEL_STUDIO_HOST = "http://localhost:8080"
LABEL_STUDIO_ACCESS_TOKEN = "7ed6ee4d3f6f0e512251bd9a9c52cd81eb302a14"


def get_project_id_from_db(task_id):
    """从数据库中获取项目ID"""
    project_id = None
    try:
        with sqlite3.connect(os.path.join(LABEL_STUDIO_DATA_PATH, "label_studio.sqlite3"),
                             check_same_thread=False) as conn:
            cursor = conn.cursor()
            # 使用参数化查询以避免SQL注入
            cursor.execute("select project_id from task where id = ?", (task_id,))
            result = cursor.fetchone()
            if result:
                project_id = result[0]
            cursor.close()
    except Exception as e:
        logging.error(f"Error fetching project ID from DB for task ID {task_id}: {e}")
    return project_id


def get_project_from_ls(project_id):
    """通过Label Studio SDK获取项目"""
    try:
        ls = label_studio_sdk.Client(LABEL_STUDIO_HOST, LABEL_STUDIO_ACCESS_TOKEN)
        project = ls.get_project(id=project_id)
        return project
    except Exception as e:
        logging.error(f"Error fetching project from Label Studio for project ID {project_id}: {e}")
        return None


if __name__ == "__main__":
    # 假设task_id是一个有效的整数
    task_id = 2
    project_id = get_project_id_from_db(task_id)

    # 检查是否成功获取了project_id
    if project_id:
        project = get_project_from_ls(project_id)
        # 这里可以处理获取到的project，例如打印信息等
        print(f"Project: {project}")
    else:
        logging.info(f"No project found for task ID {task_id}")
