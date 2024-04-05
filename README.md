# Garbage_Classification
1- In our Model we preparing dataset for Garbage classification , divided it into training and validation sets. <br />
2- We use ImageDataGenerator for data augmentation to enhance our model's ability to generalize. <br />
3- Then we try different model as base model (MobileNet , Xception) for feature extraction, and we add a global average pooling layer to reduce the spatial dimensions. <br />
&emsp;&emsp;Features that we extracted from the pre-trained model are for both training and testing datasets.
4- Then we perform feature selection using a genetic algorithm. 
&emsp;&emsp;&emsp;a) We initialize a population of binary encoded feature vectors, this feature vector representing the presence or absence of features. <br />
&emsp;&emsp;&emsp;b) Our Fitness function is evaluated using a support vector machine (SVM) classifier on the selected features.<br />
&emsp;&emsp;&emsp;c) we select parents based on their fitness. We randomly select two indices for each member of the population, compare their fitness, and select the one with higher fitness as a parent. <br />
&emsp;&emsp;&emsp;d) After that we perform a single-point crossover method between two parents that we selected to create two children. <br />
&emsp;&emsp;&emsp;e) We randomly select a crossover point and combine the genetic information of the two parents before and after that point. <br />
&emsp;&emsp;&emsp;f) And then apply mutation, and finally generate a new populations iteratively.<br />
5- After Selecting best features that suit our data we apply SVM to get final results.<br />

