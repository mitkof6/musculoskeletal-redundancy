#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import logging
# import os
# import sys
# from logging.handlers import RotatingFileHandler

# try:
#     LOG_FILENAME = os.path.splitext(__file__)[0] + ".log"
# except:
#     LOG_FILENAME = os.getcwd() + "/logging.log"


class Singleton(object):
    """
    Singleton interface:
    http://www.python.org/download/releases/2.2.3/descrintro/#__new__
    """
    def __new__(cls, *args, **kwds):
        it = cls.__dict__.get("__it__")
        if it is not None:
            return it
        cls.__it__ = it = object.__new__(cls)
        it.init(*args, **kwds)
        return it

    def init(self, *args, **kwds):
        pass


class LoggerManager(Singleton):
    """
    Logger Manager.
    Handles all logging files.
    """

    def init(self, loggername):
        logging.basicConfig(
            format='[%(asctime)s] [%(levelname)s] [@%(name)s] # %(message)s',
            # format='[%(asctime)s] [@%(filename)s:%(lineno)ds] [%(levelname)-8s] # %(message)s',
            datefmt='%F %H:%M:%S',
            # filename='example.log',
            level=logging.DEBUG)
        self.logger = logging.getLogger(loggername)

        # rhandler = None
        # try:
        #     rhandler = RotatingFileHandler(
        #         LOG_FILENAME,
        #         mode='a',
        #         maxBytes=10 * 1024 * 1024,
        #         backupCount=5
        #     )
        # except:
        #     raise IOError("Couldn't create/open file \"" +
        #                   LOG_FILENAME + "\". Check permissions.")

        # self.logger.setLevel(logging.DEBUG)
        # formatter = logging.Formatter(
        #     fmt='[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)-8s] %(message)s',
        #     datefmt='%F %H:%M:%S'
        # )
        # rhandler.setFormatter(formatter)
        # self.logger.addHandler(rhandler)

    def debug(self, loggername, msg):
        self.logger = logging.getLogger(loggername)
        self.logger.debug(msg)

    def error(self, loggername, msg):
        self.logger = logging.getLogger(loggername)
        self.logger.error(msg)

    def info(self, loggername, msg):
        self.logger = logging.getLogger(loggername)
        self.logger.info(msg)

    def warning(self, loggername, msg):
        self.logger = logging.getLogger(loggername)
        self.logger.warning(msg)


class Logger(object):
    """
    Logger object.
    """

    def __init__(self, loggername="root"):
        self.lm = LoggerManager(loggername)  # LoggerManager instance
        self.loggername = loggername  # logger name

    def debug(self, msg):
        self.lm.debug(self.loggername, msg)

    def error(self, msg):
        self.lm.error(self.loggername, msg)

    def info(self, msg):
        self.lm.info(self.loggername, msg)

    def warning(self, msg):
        self.lm.warning(self.loggername, msg)


class TestLogger(unittest.TestCase):

    def test_creation(self):
        logger = Logger()
        logger.debug("this test")


if __name__ == '__main__':
    unittest.main()
