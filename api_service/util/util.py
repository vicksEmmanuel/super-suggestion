import os

def load_env():
    """
    Load environment variables from the .env file
    """
    # Get the absolute path of the directory where the util.py file is located
    util_dir_path = os.path.dirname(os.path.realpath(__file__))
    
    # Get the absolute path of the parent directory (one level above util)
    parent_dir_path = os.path.dirname(util_dir_path)
    
    # Construct the absolute path of the .env file
    env_path = os.path.join(parent_dir_path, '.env')
    
    # Open .env file
    with open(env_path, 'r', encoding='utf-8') as f:
        # For each line in the file
        for line in f:
            # If the line is not empty and does not start with a '#'
            if line.strip() and not line.startswith('#'):
                # Split the line into key and value
                key, value = line.strip().split('=', 1)
                # Set the environment variable
                os.environ[key] = value