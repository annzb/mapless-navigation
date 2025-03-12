from pprint import pprint


class Logger:
    def __init__(self, print_log=True, loggers=tuple()):
        self.print_log = print_log
        self.loggers = loggers

    def init(self, **kwargs):
        for logger in self.loggers:
            logger.init(**kwargs)

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
