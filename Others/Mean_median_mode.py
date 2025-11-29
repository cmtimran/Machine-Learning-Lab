import numpy as np
from scipy import stats

numbers = [1, 2, 3, 4, 5, 6,6, 7, 8, 9, 10]

sum = np.sum(numbers)
count = len(numbers) 
sort = np.sort(numbers)

mean = np.mean(numbers)
median = np.median(numbers)
mode = stats.mode(numbers, keepdims=True)


print("Input numbers are: " + str(numbers) + 
"\nSorted numbers are: " + str(sort) + 
"\nSum of the numbers is: " + str(sum) + 
"\nCount of the numbers is: " + str(count) + 
"\nMeans/Average of the numbers is: " + str(mean) +
"\nMedian of the numbers is: " + str(median) +
"\nMode of the numbers is: " + str(mode.mode[0]) + " (appears " + str(mode.count[0]) + " times)")
