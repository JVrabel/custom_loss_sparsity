
import subprocess

# List of lambda values you want to experiment with

lambda_values = [0.001, 0.0005, 0.0001]
# lambda_values = [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005]
# lambda_values = [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

# Number of trials for each lambda value
num_trials = 2  # Feel free to change this

# Dictionary to hold metrics for each lambda value
all_metrics = {}

# Run the training script with each lambda value
for lambda_val in lambda_values:
    for trial_num in range(1, num_trials + 1):  # trial_num starts from 1
        # Execute the training script, passing in lambda and trial number as command line arguments
        subprocess.run(['python', 
                        'src/train.py', 
                        '--reg_lambda', str(lambda_val), 
                        '--reg_type', 'sparseloc', 
                        '--save_model', 'True', 
                        '--trial', str(trial_num),
                        '--epochs', '300'
                        ])