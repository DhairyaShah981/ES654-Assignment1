
<h3> Theoretical time complexity of the code </h3> 

Symbols Used:

1. $M$ - number of attributes in the tree construction input
2. $N$ - total number of samples in the tree construction input
3. $N'$ - total number of samples in the tree prediction input	
4. $X$ - the product of the number of discrete values in each column in the tree construction input 

<h4> Construction of the decision tree:  </h4>

The time complexity of the program can be different based on the input types and can be different based on whether we give the max_depth value or not. Let us assume that the input set has 

- Case one: The input set contains real values. 

	Now, the time complexity of the information gain function is - 
	If the function is given an input series of size N, it will have to calculate the information
	gain for all the possible splits and return the split with the maximum information gain.
	For a given split the function takes an $O(N)$ time to calculator the information gain for an 
	Input size of N. Now since there are a total of $(N-1)$ possible splits. Thus the function 
	has a time complexity of $O(N^2)$.

	Now in each of the iterations we will have to iterate through each of the column, pass it 
	to the information gain function and find the column which gives the highest information gain up on the split. Since there are M columns, in the worst case
	scenario we will have to pass all of the M columns to the function in each of the recursions. Additionally, there can be a total of N iteration in the worse
	case scenario. 
- Case two: The input set contains discrete values.

	Now the time complexity of the information gain function is - 
	If the function is given an input series of size N, it will have to go through the entire dataset an order of $N$ times and thus the complexity is $N$ in this
	case. Since each column is passed through the information gain function, each of the recursions will take order $O(NM)$ times in the worst case scenario. 

	Finding  the number of iterations: 

	- In the case that the product of all unique attributes from each of the columns is lesser than the sample size($N$):
		The total number of all possible combinations for a row in the input set is equal to the product of all the unique attributes from each of the columns.
		Thus, in the case that this value is less than the total number of samples, the maximum possible iteration will be one less than that product value
		even in the worst case scenario. 

	- In the case that the product of all unique attributes from each of the columns is greater than the sample size($N$):
	In such a case, since the decision tree is built in such a way that every non leaf node has at least two children, the maximum number of nodes and hence the
	number of recursions is equal to $(N-1)$. Thus in the worst case worst case scenario, the code will through a total of $(N-1)$ iterations. 

Thus, the time complexity for constructing the tree is:
1. Continous Input : $O(N^3M)$
2. Discrete inputs : $O(min(X,N)MN)$


<h4> Prediction of the output:  </h4>

Output for a given tree is determined by executing a depth-first search on it. Except in this case, we need not go through every node in the worst-case scenario since the conditions are specified in each node. Thus, the complexity of prediction is equal to the depth of the decision tree. Now takes the following cases: 

1. Continous input - 
	In this case, the number of attributes does not restrict the depth of the decision tree. However, the depth of the tree can not exceed $(N-1)$. Thus, the depth is of order $O(N)$
2. Discrete input - 
	In this case, the depth of the tree is restricted by both N and M. It cant exceed M since any given link cant have two of the same attribute conditions. 
	
Thus, each search will take $O(N)$ in the case of continuous inputs and $O(min(N,M))$ in the case of discrete inputs. Since the prediction input data has $N'$ samples, the final prediction complexity would be:
1. Continous Input : $O(NN')$
2. Discrete inputs : $O((M+N)N')$

<h3> Practical results: </h3> 

<u>Plots:<u/> 
1. Real input, Real output:
	- for constructing the tree: 
	
	![image](https://user-images.githubusercontent.com/76472249/214228694-8186ca01-6738-45f5-b4b9-9085debcfe2a.png)
	
	- for predicting the output: 
	
	![image](https://user-images.githubusercontent.com/76472249/214228745-21e7af18-3530-4825-a2a6-ab9d7d3a0479.png)
	
2. Real input, Discrete output:
	- for constructing the tree: 
	
	![image](https://user-images.githubusercontent.com/76472249/214228818-38fd81a1-e750-4598-8ce4-12b1f50a43ca.png)
	
	- for predicting the output: 
	
	![image](https://user-images.githubusercontent.com/76472249/214228858-d8e5b4a6-2624-433e-92ae-cdae36dc0cff.png)
	
3. Discrete input, Real output:
	- for constructing the tree:
	
	![image](https://user-images.githubusercontent.com/76472249/214229049-9fa3cf35-df27-45ae-b621-dbbda7e1c808.png)
	
	- for predicting the output:
	
	![image](https://user-images.githubusercontent.com/76472249/214229097-b2375782-6235-4829-bcc3-2695eb81b37b.png)
	
4. Discrete input, Discrete output:
	- for constructing the tree: 
	
	![image](https://user-images.githubusercontent.com/76472249/214228950-feb4c093-0920-417f-943c-a3e1a5bf6e2a.png)
	
	- for predicting the output: 
	
	![image](https://user-images.githubusercontent.com/76472249/214228981-994111c8-adde-480d-a1bf-10133cd0a523.png)

	
<h3> Comparing the experimental time complexity with the one obtained theoretically:</h3> 
	
- For construction: 
	We can clearly see that the time taken increases with the increase in $M$ and $N$ in both real and continuous inputs.
	
	1. Real inputs: 
	While theoretically, the complexity increases largely with the value of N, this will be only true for very large datasets. In smaller datasets, the time taken for each iteration will be small, and the divisions will also largely depend on the dataset. The theoretical complexity is just the upper bound. Hence, the time does not increase according to the complexity found. However, with an increase in N for large enough values of M, we can observe a sharp rise in the time taken.  

	2. Discrete Inputs:
	As mentioned above, the small size of the dataset can make the outcome non-deterministic. However, we can observe that for very large values of $N$, there is an exponential rise in the time complexity for even a small increase in $M$. This increase is gradual for smaller values of $N$. 

- For prediction:
	
	While we observed an all-most uniform increase in the time in the case of the construction of the decision tree, this was not true for prediction. This is because the time taken to predict the output only depends on the depth at which the node is placed. This value varies largely with the data, and only an upper bound can be determined. Thus, in the resultant graph, while we see an increase in the density on the plot with an increase in $N$ and $M$, this is not uniform and largely scattered.
	
