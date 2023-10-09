import logging
import time

#logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(message)s')  #基本设置，可以替代注释里的配置项
logger = logging.getLogger(__name__)   #创建logger对象
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(r'C:\Users\1\Desktop\通用python代码\log.txt')   #创建handler对象，保存文件
handler.setLevel(logging.INFO)

#log_format = '%(message)s'
log_format = "['%(asctime)s', '%(message)s']"
formatter = logging.Formatter(log_format)
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("Start print log")
logger.debug("Do something")
logger.warning("Something maybe fail.")
logger.info("Finish")