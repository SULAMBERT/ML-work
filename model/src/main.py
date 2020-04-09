from mLModel import *
import time

start_time = time.time()

#unnamed 0 column not taken into account with index_col=0
data = pd.read_csv("data/data_cleaning/final_data.csv",index_col=0)

# print(data.type.value_counts())
# data = data[:100000]
data = data.dropna()
data = data.drop_duplicates()
print(data.type.value_counts())

mlmodel = mLmodel()
mlmodel.fit_model(data)

print("--- %s seconds ---" % (time.time() - start_time))
