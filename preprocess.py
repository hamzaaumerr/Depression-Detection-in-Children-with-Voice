import csv
import os

def process():
    # Specify the path to the results directory
    results_dir = './results'
    normal_mfcc_path = os.path.join(results_dir, 'NormalMFCC.csv')
    stressed_mfcc_path = os.path.join(results_dir, 'StressedMFCC.csv')
    dataset_path = os.path.join(results_dir, 'dataset.csv')

    # Open NormalMFCC.csv and StressedMFCC.csv from the results folder
    with open(normal_mfcc_path, 'r') as normal_file, open(stressed_mfcc_path, 'r') as stressed_file:
        normal_reader = csv.reader(normal_file)
        stressed_reader = csv.reader(stressed_file)

        # Save dataset.csv in the results folder
        with open(dataset_path, 'w', newline='') as dataset_file:
            dataset_writer = csv.writer(dataset_file)
            for row in normal_reader:
                dataset_writer.writerow(row)
            for row in stressed_reader:
                dataset_writer.writerow(row)