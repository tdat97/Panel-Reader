import logging

logger = logging.getLogger()
logger_level_types = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", ]
logger.setLevel(logging.DEBUG)

# log 출력
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# log를 파일에 출력
file_handler = logging.FileHandler('log.txt')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def switch_logger_level(level="INFO"):
    """
    Level Switching Function for declared logger as global value.
    
    Level Types:
        DEBUG    : 자세한 정보
        INFO     : 확인용 정보
        WARNING  : 작은 문제 발생이지만 정상동작
        ERROR    : 함수를 실행하지 못 할 정도 문제
        CRITICAL : 프로그램이 동작하지 못할 정도 문제
    
    Function:
        Args:
            level (str): log level.
        Retruns:
            None
    """
    
    assert level in logger_level_types
    logger.setLevel(getattr(logging, level))