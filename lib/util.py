from pandas import DataFrame

def standardization(DataFrame: DataFrame):  # 標準化

    DataFrame = ((DataFrame.T - DataFrame.T.mean()) /
                 DataFrame.T.std(ddof=0)).T
    return DataFrame
