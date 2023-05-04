import seaborn as sns


# Seaborn Datasets:
# ['anagrams', 'anscombe', 'attention', 'brain_networks', 
# 'car_crashes', 'diamonds', 'dots', 'dowjones', 'exercise', 
# 'flights', 'fmri', 'geyser', 'glue', 'healthexp', 'iris', 
# 'mpg', 'penguins', 'planets', 'seaice', 'taxis', 'tips', 
# 'titanic']

# to pull and store them:
# print(sns.get_dataset_names())
diamond=sns.load_dataset("titanic", cache=True, data_home=None)
diamond.to_csv("titanic.csv", index=False)