import FileUtils
import pandas as pd
import Utils

def get_feature_risks(risks):
    feature_risks = risks.sum(axis=1)

    return feature_risks.values.tolist()

def get_individual_risks(risks):
    individual_risks = risks.sum(axis=0)

    return individual_risks.values.tolist()

if __name__ == "__main__":
    logger = FileUtils.logger(__name__, "CIG", '/mnt/c/data/')
    filenames = FileUtils.get_mbs_files()
    for filename in filenames:
        logger.log(f"Opening {filename}")
        data = pd.read_parquet(filename)

        logger.log("Calculating risks")
        risks = Utils.get_risks(data)

        logger.log("Aggregating risks")
        feature_risks = get_feature_risks(risks)
        individual_risks = get_individual_risks(risks)

        logger.log("Determining risk profile")
        risk_descriptions = []
        for risk_type in [feature_risks, individual_risks]:
            minimum = risks[risk_type].min()
            maximum = risks[risk_type].max()
            q1 = risks[risk_type].quantile(0.25)
            q3 = risks[risk_type].quantile(0.75)
            avg = risks[risk_type].mean()
            med = risks[risk_type].median()
            tup = [minimum, q1, avg, med, q3, maximum]
            risk_descriptions.append(tup)

        logger.log("Writing risk to file")
        for i in [0, 1]:
            names = ['Feature', 'Individual']
            df = pd.DataFrame(risk_descriptions[i])
            df.to_csv(df, f'{names[i]} Risks')


        break