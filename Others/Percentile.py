import numpy as np
#What is a Percentile ? #A percentile is a measure used in statistics indicating the value below which a given percentage of observations in a group of observations falls.
#What is a Quartile ? #A quartile is a measure used in statistics indicating the value below which a given percentage of observations in a group of observations falls.
#What is a Decile ? #A decile is a measure used in statistics indicating the value below which a given percentage of observations in a group of observations falls.

#Problem: calculate 90% marks over the student data [3.4, 3.4, 3.4, 3.4, 4, 3.5, 2, 1, 2.5]
scores = [3.4, 3.4, 3.4, 3.4, 4, 3.5, 2, 1, 2.5]
percentile = np.percentile(scores, 98)
print("Percentile: "+str(percentile))

#quartile
quartile = np.percentile(scores, 75)
print("Quartile: "+str(quartile))

#decile
decile = np.percentile(scores, 50)
print("Decile: "+str(decile))

 