# .......... added by @ Samadarshi ...... Please email him if there is an issue .........
import subprocess
import logging

def get_nvidia_info():
    '''
    Check for the GPU being detected correctly 
    Logs the output of nvidia-smi
    '''
    # set the logger 
    logger = logging.getLogger(__name__)

    try:
        output = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
        logger.info("....... NVIDIA-SMI Output ......\n%s", output)

        # Parse GPU name and driver version 
        lines = output.split('\n')
        for line in lines:
            if "Driver Version" in line and "CUDA Version" in line:
                logger.info("Driver and CUDA Versions: %s", line.strip())
            elif "GPU" in line and "|" in line:
                logger.info("GPU Info Line: %s", line.strip())
    
    # for command process failure 
    except subprocess.CalledProcessError as e:
        logger.error("nvidia-smi command failed: %s", e)
        
    # for not finding the drivers    
    except FileNotFoundError:
        logger.error("nvidia-smi not found. NVIDIA drivers might not be installed.")
        