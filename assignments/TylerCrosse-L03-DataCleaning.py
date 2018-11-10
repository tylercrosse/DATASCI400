"""
Lesson 03 Assignment

While it's great that the data team will back up your data, it's no good that the 
data may be faulty with outlier values and missing numbers. Spend some time 
preparing your data by identifying and dealing with the aberrant data.

Create a new Python script that includes the following:

1. Import statements for necessary package(s)
2. Create 3 different numpy arrays with at least 30 items each
3. Write function(s) that remove outliers in the first array
4. Write function(s) that replace outliers in the second array
5. Write function(s) that fill in missing values in the third array
6. Comments explaining the code blocks
7. Summary comment block on how your dataset has been cleaned up

Referenced [Function Annotations](https://www.python.org/dev/peps/pep-3107/)
"""

import numpy as np

first = np.array([2, 1, 1, 99, 1, 5, 3, 1, 4, 3,
                  101, 0, 1, 4, 8, 9, 3, 2, 1, 4,
                  175, 3, 1, 4, 2, 0, 4, 3, 4, 1])
second = np.array([2, 1, 1, 99, 1, 5, 3, 1, 4, 3,
                   101, 0, 1, 4, 8, 9, 3, 2, 1, 4,
                   175, 3, 1, 4, 2, 0, 4, 3, 4, 1])
third = np.array([2, 1, 1, 99, 1, 5, 3, "", 4, 3,
                  101, 0, "", 4, 8, 9, 3, 2, 1, 4,
                  175, 3, 1, 4, 2, 0, 4, "", 4, 1])


def remove_outliers(values: "np.array[int64]") -> "np.array[int64]":
    """
    Remove outliers using boolean mask with limits of 2 std deviantions from mean
    """
    LimitHi = np.mean(values) + 2*np.std(values)
    LimitLo = np.mean(values) - 2*np.std(values)
    FlagGood = (values >= LimitLo) & (values <= LimitHi)
    return values[FlagGood]


def replace_outliers(values: "np.array[int64]") -> "np.array[int64]":
    """
    Replace outliers with the median of non-outliers
    uses boolean mask with limits of 2 std deviantions from mean
    """
    LimitHi = np.mean(values) + 2*np.std(values)
    LimitLo = np.mean(values) - 2*np.std(values)
    FlagBad = (values < LimitLo) | (values > LimitHi)
    values[FlagBad] = np.median(values)
    return values


def fill_missing_values(values: "np.array[str672]") -> "np.array[int64]":
    """
    Fills values that are not digits with zero then recasts np.array as an integer
    """
    FlagGood = np.array([element.isdigit() for element in values])
    FlagBad = ~FlagGood
    values[FlagBad] = 0
    return values.astype(int)


"""
The frist array had any outliers removed. The removal was done with a boolean 
mask with limits of 2 standard deviations from the mean.
The second array had any outliers replaced with the median of the array. The
replacement was done using a boolean mask with limits of 2 standard deviations
from the mean.
The third array had any non digit values filled with zeros. This was done by
first constructing a boolean mask by iterating over the array to find non digits.
Then using that boolean mask to fill in the bad values.
"""
print("First Array:\n", first)
print("First Array Clean:\n", remove_outliers(first))
print("Second Array:\n", second)
print("Second Array Clean:\n", replace_outliers(second))
print("Third Array:\n", third)
print("Third Array Clean:\n", fill_missing_values(third))
