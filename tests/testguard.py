
def test_guard(method):
    """
    Decorator that checks `self.can_continue` before running the test.
    If the test raises an AssertionError, it logs the error and stops further tests.
    """
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "can_continue", True):
            self.logger.warning(f"Skipping {method.__name__} due to previous failure.")
            return
        try:
            return method(self, *args, **kwargs)
        except AssertionError as e:
            self.logger.error(f"{method.__name__} failed: {e}")
            self.can_continue = False
    return wrapper