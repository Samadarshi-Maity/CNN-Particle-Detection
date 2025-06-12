# .......... added by @ Samadarshi ...... Please email him if there is an issue .........
import logging
from datetime import datetime

# define a function for starting the logging operation 
def setup_logger(log_path, exp_name="Experiment", log_to_console=False):
    """
    Sets up a logger with a timestamped filename .
    
    Params:
        log_path: Path to the diretory of the log file.
        exp_name: Prefix for the run type .
        log_to_console: Boolean for log_to_console_enabled or disabled

    Returns:
        log_path: The name of the log file.
    """
        
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = log_path + '\\'+ f"{exp_name}_{timestamp}.log"
    
    # to make it compatible to Jupyter Notebook
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # for enabling the log_to_console options 
    handlers = [logging.FileHandler(log_filename)]
    if log_to_console:
        handlers.append(logging.StreamHandler())
    
    # start the logging operation and record in the log file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    logging.info("************* Logger Initialized ********************")
    
    return f"{exp_name}_{timestamp}.log"
