# Desciption
This is Git Repository for creating experiment for Homework 4 of CSE 431, Algorithm Engineering

## Problem 1

Merge sort has an expected run time of $\Theta(n log n)$; insertion sort has an expected run time of $\Theta(n^2)$. As such, we know that Merge sort will be faster for very large n.  Insertion sort, however, turns out to be faster for very small n.  Your job is to figure out how small.

Compare implementations of Merge sort and Insertion sort, testing each over a range of values for n. Provide a graph of the results, clearly indicating the value on n where the lines cross (or a range of values where they essentially overlap).

You may use your own implementations, in any language you choose, or ones that you find elsewhere, as long as you cite your sources.  You must use a wide enough range of values of n to provide a convincing argument of your answer.


#### Experiment code

Experiment code will measure average running time of merge sort and insertion 
sort on several sizes of input and create figure called figure.png to show the result.  
  

Run the experiment code by
```
python HW4_Q1.py
```

## Problem 2  
Given the results of part 1 (Merge sort vs. Insertion sort), use your implementations to build a hybrid sorting algorithm that combines the two (called Tim sort).  When you recurs in Merge sort, if a partition size is less than or equal to some constant k, you should use insertion sort, but if it is greater than k you should continue with Merge sort.  Experimentally determine what value of k will optimize speed.

Is k the same as the crossover point for part 1?  Why do you think this is or is not the case?  Generate a graph comparing this hybrid sort to both Insertion Sort and Merge sort.  Make sure to test with a wide range of values for k (to be sure of your answer) as well as more than one size of n (in case the number of values sorted affects your answer).
#### Experiment code

Experiment code will measure average running time of merge sort, insertion 
sort, and hybrid sort on several sizes of input and create figure called figure.png to show the result.  
  

Run the experiment code by
```
python HW4_Q2.py --include_insertion True --num_trial 100
python HW4_Q2.py
```

## Problem 3  
Compare two back ends for a dictionary, such as a binary tree and a hash table. For example, in the C++ standard library, \text{std::multimap} can be used to easily store a 
dictionary of key-value pairs in a balanced binary tree, while 
\text{std::unordered\_multimap} can be used to store values in a hash table.  Other 
structures such as a sorted array could also be used as possible dictionary back ends.

In theory, a hash table should be faster for insertions and deletions, but how much 
faster?  Compare the two for insertion of n = 10, n = 100, n = 1000, …, n = as high as you 
need to go for at least one of the data structures to take more than 3 seconds to run.  

#### Experiment code

Experiment code will measure average running time of merge sort and insertion 
sort on several sizes of input and create figure called figure.png to show the result.  
  

Run the experiment code by
```
python HW4_Q3.py 
```