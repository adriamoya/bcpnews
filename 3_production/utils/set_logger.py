
import logging

def set_logger(log_name):
    """
    Sets the log file
    """
    logger = logging.getLogger(__name__)

    # INFO

    # levels
    logger.setLevel(logging.INFO)

    # create a file handler
    handler = logging.FileHandler('%s.log' % log_name)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger
