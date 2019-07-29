from math import ceil
import numpy as np

path = '/home/elm/data/preprocessed_mimic_data/'
files = [('diag_proc.csv', ["ICUSTAY_ID", "ICD9_CODE", "DAYS_TO_OUT"]), ('charts_prescriptions.csv', ["ICUSTAY_ID", "VALUECAT", "HOURS_TO_OUT"])]

def convert_to_seconds(timestamp, header):
    if header == "DAYS_TO_OUT":
        return ceil(float(timestamp) * 24 * 3600)
    elif header == "HOURS_TO_OUT":
        return ceil(float(timestamp) * 3600)
    else:
        raise ValueError("Not a valid header")

for (name, header) in files:
    output_file_path = f"{name}_output.txt"
    with open(path + name, 'r') as input_file, open(output_file_path, 'w+') as output:
        current_id = None
        current_time = None
        for each_line in input_file:
            row = [x.strip() for x in each_line.split(',')]
            if row[0] != current_id:
                current_id = row[0]
                current_time = convert_to_seconds(row[2], header[2])
                output.write(f"{line}]]]\r\n")
                line = f"{current_id}, [[{current_time}, [{row[1]}"
            else:
                new_time = convert_to_seconds(row[2], header[2])
                if new_time != current_time:
                    current_time = new_time
                    line = f"{line}], {current_time}, [{row[1]}"
                else:
                    line = f"{line}, {row[1]}"
        
        output.write(f"{line}]]]\r\n")


