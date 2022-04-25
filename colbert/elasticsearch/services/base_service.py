import logging

from colbert.elasticsearch.config.config import Config


class BaseService:
    def __init__(self, config: Config):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_retry = 3
        self.config = config

    def make_request(self, request_func, fail_call_back=None):
        done = False
        c = self.max_retry
        output = None
        while not done and c > 0:
            try:
                output = request_func()
                done = True
            except Exception as e:
                c -= 1
                self.logger.error(f"Cannot make request. Error: {e}")
                print(f"Cannot make request. Error: {e}")
                if fail_call_back is not None:
                    fail_call_back()

        return output
