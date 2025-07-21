import random
import string
from datetime import datetime
from pprint import pprint


class Logger:
    def __init__(self, print_log=True, loggers=tuple(), run_name=None):
        self.print_log = print_log
        self.loggers = loggers
        self._run_name = None
        self._user_run_name = run_name

    def init(self, **kwargs):
        logger_run_name = None
        for logger in self.loggers:
            logger.init(**kwargs)
            if logger_run_name is None and getattr(logger, 'name', None):
                logger_run_name = logger.name
        
        if self._user_run_name:
            self._run_name = self._user_run_name
        elif logger_run_name:
            self._run_name = logger_run_name
        else:
            date_str = datetime.now().strftime("%d%b%y").lower()
            rand_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
            self._run_name = f"{date_str}_{rand_id}"

    def run_name(self):
        return self._run_name

    def log(self, stuff):
        if self.print_log:
            if isinstance(stuff, str):
                print(stuff)
            else:
                pprint(stuff)
        if isinstance(stuff, dict):
            for logger in self.loggers:
                logger.log(stuff)

    def finish(self, **kwargs):
        for logger in self.loggers:
            logger.finish(**kwargs)
