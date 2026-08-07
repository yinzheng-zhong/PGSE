import os

from pgse.log.logger import Logger as _Logger

# Logging goes to the console only. Set PGSE_LOG_FILE, or call
# logger.add_file_handler(path), to also append it to a file.
logger = _Logger('pgse', log_file=os.environ.get('PGSE_LOG_FILE'))
