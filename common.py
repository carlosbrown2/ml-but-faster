import time

def timed_evaluation(func):
    """Decorator to time the execution of a function and return the coefficients."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        betas = func(*args, **kwargs)  # Call the decorated function
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time for {func.__name__}: {elapsed_time:.4f} seconds")
        return betas  # Return the coefficients (betas)
    return wrapper
