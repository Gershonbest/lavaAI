import logging

class Logger:
    def __init__(self, level=logging.INFO, format="%(levelname)s - %(asctime)s - %(message)s"):
        self.logger = logging.getLogger(__name__)
        if not self.logger.hasHandlers():  # Prevent duplicate handlers
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(format))
            self.logger.addHandler(handler)
        self.logger.setLevel(level)
    
    def get_logger(self):
        return self.logger