import logging

class CustomLogger:
    def __init__(self, name, log_level=logging.INFO, log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        formatter = logging.Formatter(log_format)

        # Create a console handler and set the formatter
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)

        # Add the console handler to the logger
        self.logger.addHandler(ch)

    def getLogger(self):
        return self.logger
