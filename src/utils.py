import logging

# -------------------------
# Setup Logging
# -------------------------
def setup_logging(log_file='pipeline.log'):
    """
        Sets up the logging configuration for the application.

        This function configures the logging module to log messages to both a file and the console.
        By default, it logs messages to a file named 'pipeline.log'. The log format includes the
        timestamp, log level, and the log message.

        Args:
            log_file (str): The name of the log file. Default is 'pipeline.log'.

        Returns:
            None

        Example:
            setup_logging()  # Sets up logging with default log file.
            setup_logging('my_log.log')  # Sets up logging with a custom log file.
        """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging is set up.")
