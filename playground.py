import pandas as pd

def analyze_dataset(filename):
    # Load the data from the specified file
    data = pd.read_csv(filename, sep='\t')
    data['response_time'] = pd.to_numeric(data['response_time'], errors='coerce')

    # Separate the data by trial type
    one_back = data[data['trial_type'] == '1-back']
    two_back = data[data['trial_type'] == '2-back']

    # Filter correct trials and extract response times
    one_back_correct = one_back[one_back['accuracy'] == 1.0]
    one_back_correct_rt = one_back_correct['response_time'].dropna()

    two_back_correct = two_back[two_back['accuracy'] == 1.0]
    two_back_correct_rt = two_back_correct['response_time'].dropna()

    # Function to display detailed calculations
    def detailed_mean_calculation(rt_series, condition_name):
        print(f"Detailed Calculation for {condition_name} Correct Trials:")
        print("-" * 50)
        # List all response times
        print("Response Times (seconds):")
        for idx, rt in enumerate(rt_series, 1):
            print(f"Trial {idx}: {rt:.3f}")
        print("-" * 50)
        # Calculate sum and count
        total_rt = rt_series.sum()
        count_rt = rt_series.count()
        print(f"Sum of Response Times: {total_rt:.3f} seconds")
        print(f"Number of Correct Trials: {count_rt}")
        # Calculate mean
        mean_rt = total_rt / count_rt if count_rt != 0 else float('nan')
        print(f"Mean Response Time: {mean_rt:.3f} seconds")
        print("\n")
        return mean_rt

    # Perform calculations for each condition
    one_back_mean_rt = detailed_mean_calculation(one_back_correct_rt, '1-back')
    two_back_mean_rt = detailed_mean_calculation(two_back_correct_rt, '2-back')

    # Return results as a dictionary
    results = {
        'filename': filename,
        'one_back_mean_rt': one_back_mean_rt,
        'two_back_mean_rt': two_back_mean_rt
    }
    return results

# Example usage
if __name__ == "__main__":
    # List of data files to analyze
    data_files = ['participant1_data.tsv', 'participant2_data.tsv', 'participant3_data.tsv']
    all_results = []

    for file in data_files:
        print(f"Analyzing {file}...")
        results = analyze_dataset(file)
        all_results.append(results)

    # Convert results to a DataFrame for summary
    results_df = pd.DataFrame(all_results)
    print("\nSummary of Results:")
    print(results_df)

# Assuming your script is named analyze_vswm_data.py

if __name__ == "__main__":
    # Replace 'your_data_file.tsv' with the actual filename of your dataset
    results = analyze_dataset('your_data_file.tsv')
    print("Final Results:")
    print(results)




