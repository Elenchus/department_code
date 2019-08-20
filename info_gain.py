import file_utils
import pandas as pd
import Utils

import gc

def get_feature_risks(risks):
    feature_risks = risks.sum(axis=1)

    return feature_risks.values.tolist()

def get_individual_risks(risks):
    individual_risks = risks.sum(axis=0)

    return individual_risks.values.tolist()

if __name__ == "__main__":
    logger = file_utils.logger(__name__, "CIG", '/mnt/c/data/')
    filenames = file_utils.get_mbs_files()
    for filename in filenames:
        logger.log(f"Opening {filename}")
        data = pd.read_parquet(filename)

        logger.log("Calculating risks")
        risks = Utils.get_risks(data)

        del data
        gc.collect()

        logger.log("Writing risks to file")
        risks.to_csv("Risks_file.csv")

        logger.log("Aggregating risks")
        feature_risks = get_feature_risks(risks)
        individual_risks = get_individual_risks(risks)

        del risks
        gc.collect

        logger.log("Determining risk profile")
        risk_descriptions = []
        for risks in [feature_risks, individual_risks]:
            minimum = risks.min()
            maximum = risks.max()
            q1 = risks.quantile(0.25)
            q3 = risks.quantile(0.75)
            avg = risks.mean()
            med = risks.median()
            tup = [minimum, q1, avg, med, q3, maximum]
            risk_descriptions.append(tup)

        logger.log("Writing results to file")
        for i in [0, 1]:
            names = ['Feature', 'Individual']
            df = pd.DataFrame(risk_descriptions[i])
            df.to_csv(df, f'{names[i]} Risks')

        break