# Assignment 4

## Instructions
* In the KNN implementation, extract the nearest_model.zip file before testing the algorithm. The .txt was too large which is why we chose to compress it.
* For any index out of range errors, kindly check the last line of the .txt file for any potential white spaces/new lines.

## K-Nearest Neighbours

### Problem Statement
We are given an image dataset and are required to classify the orientation of the given image. The possible orientations are 0, 90, 180, 270.

### Solution Formulation
* Load the image data by splitting the File IDs, Labels and RGB Pixel Values
* Initialize K to your chosen number of neighbors
* For each example in the data
   1. Calculate the distance between the query example and the current example from the data.
   2. Add the distance and the index of the example to an ordered collection
* Sort the ordered collection of distances and indices from smallest to largest (in ascending order) by the distances
* Pick the first K entries from the sorted collection
* Get the labels of the selected K entries

### Implementation
* Once the input test/train files are read, I created functions for minmax normalization that change the 0-255 pixel range down to 0-1. This helped me with certain consistency issues that I was facing before.
* The model.txt file comprises of file label along with it's normalized RGB values for testing purposes.
* The euclidean distance function computes the summation of distances between the query file and each row in the model file to generate a list with the final distance values.
* A nearest neighbour function is defined that sorts the distance list in ascending order to get the closest images to the test file near the top. Based on the value of K, the top 'K' values are considered as the nearest neighbours and returned in a neighbour list.
* Prediction function takes the neighbour list and finds the maximum count of labels associated with the best distances and returns it as the prediction.
* The accuracy can be checked by comparing the predicted values with the actual, given labels to compute efficiency of the algorithm.

### Design choices and problems
* We wanted to implemented a brute force approach to KNN to show the true conceptual nature of the algorithm (which happens to be very slow, because each image vector is compared with every image vector in the mode file), which is why we chose to use as vanilla a python code as opposed to a scikit approach.
* After tinkering with the value 'K' (nearest neighbours), we chose to use K = 10 as our value which gave us a better accuracy than some of the other values.
* We chose a minmax normalization approach to bring down the 0-255 values which gave us slightly better and consistent results.

### Experimentation Results
* Using about half the training data for training gave us a training accuracy of about 76.92% and testing the algorithm over the entire test file gave us an accuracy of 70.83%. The algorithm however still remains quite slow due its brute force nature and the lack of helper functions (like cdist as a highly time efficient euclidean distance computation).

#### Correctly classified images

![alt text](10008707066.jpg "Correct classification") 
![alt text](10099910984.jpg "Correct classification")
![alt text](10107730656.jpg "Correct classification")
![alt text](10161556064.jpg "Correct classification")

#### Incorrectly classified images

![alt text](10196604813.jpg "Correct classification") True: 90 Predicted: 0

![alt text](10351347465.jpg "Correct classification") True: 270 Predicted: 180

![alt text](10352491496.jpg "Correct classification") True: 90 Predicted: 180

![alt text](10484444553.jpg "Correct classification") True: 180 Predicted: 0


## Decision Tree Model:

### General plan:
While a Decision Tree Classification is rather intuitive, there is still the matter of picking the right feature and numeric value to divide the continuous data to decide on a label. To determine these criterion Information gain and Entropy is used, where the chosen unique pixel value corresponding to a certain feature returns the maximum ‘information gain’. Information Gain is given by:
"Information Gain = Entropy(parent) – Weighted Average Entropy(children)"  where
"Entropy= Sum( - p_i * log2(p_i))" where p_i is the probability/proportion of a label in the dataset
Weighted Average Entropy= (proportion of true branch in original dataset) * Entropy(true branch) + (proportion of false branch in original dataset) * Entropy(false branch) 

### Steps/Functions:
* ‘create_feature_table’: Each feature corresponds to a value of colour (red, green or blue) for a certain pixel.
* 'compute_avg_disorder: Calculates the weighted average entropy of the split(children) datasets
* ‘entropy’: Calculates entropy for parent dataset.
* 	‘best_split’: Returns the best feature & best value (within unique values 0-255) on the basis which the data is split, along with the best gain which decided on this feature
* 	create_leaf: Given that the tree reaches its maximum depth or all the labels in the dataset are the same a ‘leaf’ or decision node is created.
*	check_purity: Checks if all the labels in the dataset are the same.
*	‘decision_tree’: A tree that is built by recursively splitting the data based on the best feature and best value for that dataset till a decision is reached (defined by reaching the maximum depth or having all labels of the split dataset be the same)
*	A dictionary was used to save the parameters of the decision tree created such that the key served as the parent node denoted by a tuple (best feature, best pixel value) and the values was a list of a pair of values [given the condition is true, given the condition is false]

### Problems faced:
*	Training the original dataset takes way too long (4 hours+). Since it was known that the logic of the code makes sense, a smaller train data was used to model the data. The first 1000 data points(i.e 250 images were taken to model the data). It isn’t ideal for a dataset that has 943 images but it was really taking too long irrespective of the depth change.
*	Reducing the number of features was considered but that drastically reduced the accuracy of the model and without PCA implementation through library there wouldn’t be a logic to removing features.
*	Leaf didn’t assign labels instead was labelled ‘None’: when 2 labels are equally occurring and maximum in the dataset argmax() returns the first occurring value or ‘None’. To avoid this a function was used to randomly break the tie and choose one of the most frequently occurring labels. The solution was to correct the mistake of returning the value by the leaf, there previously was no return function.
*	Took approximately 25+ minutes to run tree for 7 features. Even when trying to implement shallow max_depth of tree.

### Results:
*	Training takes too long on 32976 images. For an initial analysis, the model was trained on only 300 of training images, with a max_depth = 10, which gave a 51% accuracy on this particular testing dataset
*	A model trained on 1000 training images with a max_depth of 4, gave a 60% accuracy on this particular dataset and with a max_depth of 6 gave 59.27% accuracy
*	
Resources used to understand Decision Tree algorithm without libraries:
*	https://www.youtube.com/watch?v=y6DmpG_PtN0
*	https://www.youtube.com/watch?v=LDRbO9a6XPU&t=3s


## Neural Network

### Problem Statement
We are given an image dataset and are required to classify the orientation of the given image. The possible orientations are 0, 90, 180, 270.

### Neural Network Training Process
* We read the training data and its corresponding labels and convert it into a numpy matrix. 
* The labels are onehot encoded to represent 4 different orientations. 
* We also normalize the training data values so that the values of all neurons remain between 0 and 1. We do this normalization simply by dividing the entire numpy matrix by 255. We divide by 255 because that is the maximum possible value of a rgb for a pixel. 
* Initially, the weights are initialized randomly using a normal distribution. And biases are initialized to 0.
* The forward propagation step calculates the probabilities of each orientation for each image.
* We are using a softmax function in the output layer to compute the probabilities of each class.
* At the end of the forward propagation step, we calculate the cross-entropy loss for the network. 
* Then during the backpropagation step, we update the values of all the weights and biases in the network based on the loss computed so that we can minimize the loss in the next iteration.
* We use a hyperparameter called learning rate that defines how big or small step to take during gradient descent. Too small learning rate can lead to very slow training whereas too large learning rate can lead to overshotting.
* We do this training for a specific number of iterations and try to minize the loss.
* Once the model is trained, we save the learned parameters to a nnet_model.json file.
* During testing, we read the learned parameters from the nnet_model.json file and run a single forward propagation pass through the network to compute the predicitons on the test data. We also, write the predictions to an output.txt file.

### Defining the Network
* Learning rate = 0.1
* Number of iterations = 10000
* Number of layers = 4 (3 hidden, 1 output)
* Number of nodes in each layer = 16(layer 1), 10(layer 2), 8(layer 3), 4 (output layer)
* Activation functions = tanh (for all hidden layers), softmax (for output layer) 
* Loss function = cross-entropy loss for softmax function
* Time to train = About 10 minutes

### Design choices and problems
* We use tanh instead of sigmoid because it is considered a better activation function than sigmoid for hidden layers. We test the network using sigmoid as well, but get less accuracy. We try to use ReLU function as well, instead of the tanh function, but were unable to do so because we couldn't figure out how to calculate its derivative during backpropagation. But, we believe that using ReLU would increase the performance of the network at least slightly. It would at least lead to a faster training time for the network.
* We keep the network relatively small to get a smaller training time. We try to create a bigger network as well (# of nodes: 128(layer 1), 64 (layer 2), 32 (layer 3), 4(outpur layer)). This large network took about 3 hours to train and gave a training accuracy of 78% and test accuracy of 73%. Since the improvement wasn't signifacnt, we use the smaller network which is a bit less accurate but trains much faster. 
* The other hyperparameters were chosen after experimentation. We started by creating a network with only 1 hidden layer and then added more layers to get better results. This increased the accuracy from about 30% to 70%.
* We pick a relatively larger learning rate because we use less number of iterations. More iterations don't really lead to better results.

### Experimentation Results

| Learning rate | Number of iterations | Nodes          | Layers | Training accuracy | Test Accuracy | Running Time     |
|---------------|----------------------|----------------|--------|-------------------|---------------|------------------|
| 0.02          | 30000                | 128, 64, 16, 4 | 4      | 78                | 73            | About 3 hours    |
| 0.1           | 10000                | 16, 10, 8, 4   | 4      | 74                | 70            | About 12 minutes |
| 0.01          | 40000                | 16, 4          | 2      | 33                | 32            | About 2 minutes  |
| 0.1           | 10000                | 8, 6, 5, 4     | 4      | 72                | 67            | About 9 minutes  |

* Using about half the training data for training gives us a train accuracy of 75% and a test accuracy of 70%. The difference between these accuracies suggest the network might be overfitting.
* Using only 6000 samples from the training data to train the model leads to a training accuracy of 78% and a test accuracy of 70% which again suggests overfitting.
* If we were to use a model in production (for a client), we would most likely suggest the second model (as per the table) since it gives a good accuracy, doesn't overfit much, and also trains relatively faster. 
* As we can see form the table, a larger network doesn't lead to significantly better results. But a much smaller network leads to much worse results.

#### Correctly classified images

![alt text](10008707066.jpg "Correct classification") 
![alt text](10099910984.jpg "Correct classification")
![alt text](10107730656.jpg "Correct classification")
![alt text](10161556064.jpg "Correct classification")

#### Incorrectly classified images

![alt text](10196604813.jpg "Correct classification") True: 90 Predicted: 270

![alt text](10351347465.jpg "Correct classification") True: 270 Predicted: 180

![alt text](10352491496.jpg "Correct classification") True: 90 Predicted: 180

![alt text](10484444553.jpg "Correct classification") True: 180 Predicted: 0

## Best Model
* Best model uses our neural network from nnet.py with different parameters.
* Its hyperparameters are:
    * Learning rate = 0.02
    * No. of iterations = 30000
    * Number of layers = 4
    * Number of nodes = 128(layer 1), 64(layer 2), 16(layer 3), 4 (output layer)
* Training Accuracy = 78%
* Test accuracy = 73%
* It takes a training time of about 3 hours.
* The long training time is the reason we don't use it in the nnet.py model. This model gives a better accuracy but at the cost of high training time.

