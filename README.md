# Linear-Discriminant-Analysis-from-scratch

LDA is a dimensionality reduction technique commonly used for feature extraction and classification tasks.
The code first imports the necessary libraries, including sklearn for datasets, data manipulation, model selection, metrics, and the KNN classifier. Then, it defines a function named LDA that performs LDA on the given data X with corresponding labels y and number of components n. The function calculates the within-class scatter matrix (SW) and between-class scatter matrix (SB) based on the provided data. It then computes the transformation matrix A as the inverse of SW multiplied by SB. The function further extracts the eigenvalues and eigenvectors of A and sorts them in descending order. Finally, it returns the dot product of the data X and the first n eigenvectors, which represents the transformed data.

Next, the Iris dataset is loaded using the datasets module from sklearn, and the data is stored in a pandas DataFrame named iris_df. The code splits the data into features x and labels y. Then, it converts the DataFrame to NumPy arrays for further processing.

The code proceeds by splitting the data into training and testing sets using the train_test_split function from sklearn.model_selection. It prints the sizes of the training and test sets. After that, it creates an instance of the KNN classifier with n_neighbors=5 and fits the model on the training data. The predictions are made on the test data, and the accuracy score is calculated using accuracy_score from sklearn.metrics. The accuracy is printed, indicating the performance of KNN without LDA.

Next, the LDA function is called to transform the data x and reduce its dimensionality to 2. The transformed data is then split into training and testing sets. Another instance of the KNN classifier is created, trained on the transformed training data, and used to predict the labels of the transformed test data. The accuracy score is computed and printed, indicating the performance of KNN with LDA.

Finally, a comment suggests that using LDA as a preprocessing step increases the accuracy of KNN. However, it also notes that other factors such as the number of neighbors, the number of components, and the test size can affect the accuracy of KNN.
