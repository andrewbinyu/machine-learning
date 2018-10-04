 Load the Boston housing data from the sklearn datasets module
• Describe and summarize the data in terms of number of data points, dimensions, target, etc
• Visualization:presentasinglegridcontainingplotsforeachfeatureagainstthetarget.Choose the appropriate axis for dependent vs. independent variables. Hint: use pyplot.tight layout function to make your grid readable
• Divide your data into training and test sets, where the training set consists of 80% of the data points(chosenatrandom). Hint:Youmayfindnumpy.random.choiceuseful
• Write code to perform linear regression to predict the targets using the training data. Remem- ber to add a bias term to your model.
• Tabulate each feature along with its associated weight and present them in a table. Explain what the sign of the weight means in the third column (’INDUS’) of this table. Does the sign match what you expected? Why?
• Test the fitted model on your test set and calculate the Mean Square Error of the result.
• Suggest and calculate two more error measurement metrics; justify your choice.
• Feature Selection: Based on your results, what are the most significant features that best pre- dict the price? Justify your answer.
