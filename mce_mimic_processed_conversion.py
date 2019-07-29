from math import ceil
import numpy as np

path = '/home/elm/data/preprocessed_mimic_data/'
# files = [('charts_prescriptions.csv', ["ICUSTAY_ID", "VALUECAT", "HOURS_TO_OUT"])]
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
        line = None
        x = ",".join(header) + '\n'
        for each_line in input_file:
            if each_line == x:
                continue

            row = [x.strip('\n') for x in each_line.split(',')]
            if row[1] == '':
                continue

            if row[0] != current_id:
                current_id = row[0]
                if row[1] == '':
                    current_time = 0
                else:
                    current_time = convert_to_seconds(row[2], header[2])
                
                if line is not None:
                    output.write(f"{line}]]]\r\n")

                line = f"{current_id}, [[{current_time}, [{row[1]}"
            else:
                if row[2] == '':
                    new_time = current_time
                else:
                    new_time = convert_to_seconds(row[2], header[2])

                if new_time != current_time:
                    current_time = new_time
                    line = f"{line}], {current_time}, [{row[1]}"
                else:
                    line = f"{line}, {row[1]}"
        
        output.write(f"{line}]]]\r\n")


