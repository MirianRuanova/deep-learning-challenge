# deep-learning-challenge

## Overview of the Analysis:

The purpose of this analysis is to build a deep learning model to help Alphabet Soup, a fictional charity organization, predict whether applicants are successful in receiving funding based on various input features. By creating an accurate model, Alphabet Soup aims to make more informed decisions on which applicants to fund, optimizing their resources and maximizing the impact of their contributions.

## Results:

### Data Preprocessing:
- Target Variable: The target variable for our model is the "IS_SUCCESSFUL" column, which indicates whether an applicant was successful (1) or unsuccessful (0) in receiving funding.

- Features: The features for our model include columns like "ASK_AMT," "APPLICATION_TYPE," "AFFILIATION," "CLASSIFICATION," "USE_CASE," "ORGANIZATION," "INCOME_AMT," and "SPECIAL_CONSIDERATIONS."

- Variables to Remove: The "EIN" and "NAME" column should be removed from the input data because it serves as an identifier and does not contribute to the prediction task.

### Compiling, Training, and Evaluating the Model:
- Neurons, Layers, and Activation Functions: We selected a deep learning model with three dense layers:
    1. First hidden layer: This layer has 64 neurons (units) with the ReLU activation function. It takes the input data with a dimension of num_input_features.
    
    2. Second hidden layer: This layer has 32 neurons (units) with the ReLU activation function.

    3. Output layer: This layer has 1 neuron (unit) with the sigmoid activation function, suitable for binary classification tasks. 


- Model Performance: We aimed to achieve high accuracy (75%) on the testing dataset to predict successful funding applicants accurately. While the model provided reasonably good performance , it did not achieve the target accuracy . We observed the accuracy on the test set to be below the desired threshold (72.77%).

- Steps to Improve Model Performance: To enhance the model's performance, we experimented with the following techniques:

     1. Removing Outliers: We implemented the `drop_outliers_iqr()` function to remove outliers using the Interquartile *Range (IQR) method*. Outliers can negatively impact the model's performance by introducing noise and bias. The function calculates the IQR for the specified column and defines lower and upper bounds based on a multiplier. Rows with values outside the bounds are dropped from the dataset.

     Model Architecture: We designed a more sophisticated deep learning model with multiple hidden layers using the *Keras Sequential API*. The architecture includes four dense hidden layers with 64, 32, 16, and 8 neurons, respectively. These layers utilize the Rectified Linear Unit (ReLU) activation function to introduce non-linearity and enhance the model's ability to capture complex relationships in the data. Additionally, we included a *dropout layer* with a dropout rate of *0.2* to *prevent overfitting*. Accuracy pf 72.44%.

     2. In the second attemp we followed the same archetecture as before but we set the number of epochs to 150 and the batch size to 32. During each epoch, the model iteratively adjusts its internal parameters using the training data to minimize the loss function. The batch size defines the number of samples used in each update, and it influences the speed and stability of training.. We got an accuracy of 72.65%

     3. In the third attempt we reduced the epochs to 50. Accuracy of 72.52%.
     


## Summary:

The deep learning model showed promising results; however, it did not reach the desired accuracy on the testing dataset. It may still be useful for predicting successful applicants to some extent, but it requires further improvement to optimize funding decisions for Alphabet Soup.

To potentially solve this classification problem more effectively, we recommend exploring other models like Random Forest, Gradient Boosting, or Support Vector Machines (SVM). These models are well-suited for tabular data and can capture complex relationships between features and the target variable. Additionally, applying feature engineering techniques or further preprocessing the data, such as binning rare occurrences, may also help enhance the model's performance.

Ultimately, the choice of the best model would depend on the characteristics of the dataset and the specific requirements of Alphabet Soup. It is crucial to continue experimenting with different models and hyperparameters to find the most accurate and reliable solution for this important funding decision-making task.