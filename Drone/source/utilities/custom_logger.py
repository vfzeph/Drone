import logging
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import os
from torch.utils.tensorboard import SummaryWriter

class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    def format(self, record):
        log_record = {
            "time": datetime.utcfromtimestamp(record.created).isoformat(),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if isinstance(record.args, dict):
            log_record.update(record.args)
        return json.dumps(log_record)

class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler using ThreadPoolExecutor."""
    def __init__(self, handler):
        super().__init__(handler.level)
        self.handler = handler
        self.executor = ThreadPoolExecutor(max_workers=1)

    def emit(self, record):
        self.executor.submit(self._emit, record)

    def _emit(self, record):
        self.handler.emit(record)

    def close(self):
        self.handler.close()
        self.executor.shutdown()
        super().close()

class CustomLogger(logging.Logger):
    def __init__(self, name, log_dir="logs", log_level=logging.DEBUG):
        super().__init__(name)
        os.makedirs(log_dir, exist_ok=True)
        tensorboard_log_dir = os.path.join(log_dir, "tensorboard_logs")
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        self.setLevel(log_level)
        self.handlers = self.setup_handlers(log_dir)
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

    def setup_handlers(self, log_dir):
        handlers = []

        # Setup detailed log file handler
        detailed_log_path = os.path.join(log_dir, 'detailed.log')
        detailed_handler = logging.FileHandler(detailed_log_path)
        detailed_handler.setLevel(logging.DEBUG)
        detailed_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        async_detailed_handler = AsyncLogHandler(detailed_handler)
        self.addHandler(async_detailed_handler)
        handlers.append(async_detailed_handler)

        # Setup console handler for less detailed logs
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s: %(message)s'))
        async_console_handler = AsyncLogHandler(console_handler)
        self.addHandler(async_console_handler)
        handlers.append(async_console_handler)

        # Setup JSON file handler for structured logs
        json_log_path = os.path.join(log_dir, 'structured.json')
        json_handler = logging.FileHandler(json_log_path)
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(JsonFormatter())
        async_json_handler = AsyncLogHandler(json_handler)
        self.addHandler(async_json_handler)
        handlers.append(async_json_handler)

        return handlers

    def get_logger(self):
        """Return the initialized logger instance."""
        return self

    def close_handlers(self):
        """Close all handlers to properly release resources."""
        for handler in self.handlers:
            handler.close()
            self.removeHandler(handler)
        self.writer.close()

    def log_scalar(self, tag, value, step):
        """Log a scalar value to TensorBoard."""
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

if __name__ == "__main__":
    logger = CustomLogger("TestLogger", log_dir="./logs")
    test_logger = logger.get_logger()
    
    # Example usage of the logger
    test_logger.info("This is an info message.")
    test_logger.debug("This is a debug message.")
    test_logger.warning("This is a warning message.")
    test_logger.error("This is an error message.")
    
    # Example usage of TensorBoard scalar logging
    for step in range(100):
        logger.log_scalar("example_scalar", step * 0.1, step)
    
    # Close handlers
    logger.close_handlers()
