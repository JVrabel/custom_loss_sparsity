
import subprocess

# List of lambda values you want to experiment with
lambda_values = [0.05, 0.05, 0.02, 0.02, 0.01, 0.005, 0.005, 0.002, 0.002, 0.001, 0.001, 0.0005, 0.0005, 0.0002, 0.0002, 0.0001, 0.0001, 0.00005, 0.00005]

# Dictionary to hold metrics for each lambda value
all_metrics = {}

# Run the training script with each lambda value
for lambda_val in lambda_values:
    # Run the script. This assumes that you have added argparse to your train.py to accept lambda as a parameter
    # subprocess.run(['python', 'src/train.py',  '--reg_type', 'vanilla'])
    subprocess.run(['python', 'src/train.py', '--reg_lambda', str(lambda_val), '--reg_type', 'sparseloc'])

# 