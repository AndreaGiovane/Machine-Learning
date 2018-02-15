# Machine-Learning
The repository contains all the files to reproduce the experiment of my project.

I used Python3 with the external libraries numpy and matplotlib. Finally, I used the MNIST dataset of handwritten images to perform a binary classification of 3 and 5. To construct the model I selected 1000 samples of 3 and 1000 samples of 5. Then to test it, I used the remaining part.

Instructions to reproduce:
  - download the file 'AndreaGiovane.zip'
  - extract
  - execute file 'Lasso.py'

Description of my code:
My code is an implementation of Lasso with coordinate descent algorithm.
At each iteration I take off a variable and compute the correlation between the residue without this variable and the variable extract. If they are not correlated enough, the respective coefficient is set to zero.
The procedure continue until a convergence is reached. The convergence is achieved when the difference between the difference between the residual on the previous round and the current residual is below a certain threshold.
