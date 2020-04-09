from dataCleaningModel import *


import time
start_time = time.time()
dataclass = dataCleaning()
data = dataclass.read_data()
print(data.head())
print(data.type.value_counts())
data.to_csv("final_data_category.csv")

# data = pd.read_csv("data/data_cleaning/final_data.csv")
# print(data.type.value_counts())
# data = data.dropna()
# print(data.columns)
# print(len(data.columns))

print("--- %s seconds ---" % (time.time() - start_time))

