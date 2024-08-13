# Implementing Classification Methods to Predict Student Grades

INTRODUCTION

The educational achievement of students is a multifaceted issue influenced by various demographic, social and school-related factors. Gaining insights into these factors can offer valuable guidance for aiming to enhance students' educational experiences and outcomes. This report analyses student performance data from two Portuguese secondary schools by utilising decision tree techniques to predict academic performance in Mathematics.

DATASET INFORMATION

The dataset used in this analysis was obtained from the UCI Machine Learning Repository which comprises 33 variables containing detailed records of students' academic achievements, demographic information, social background and other relevant attributes. The dataset includes grades from three assessment periods in Mathematics (G1, G2 and G3), which are the primary variables of interest. The attributes encompass various factors, including age, gender, family background, study habits and extracurricular activities.

OBJECTIVES

The primary aim of this study is to apply machine learning models to predict student grades in Mathematics and identify the most significant factors influencing these grades. The analysis is structured into three main sections:

1. Regression Analysis to Predict G1: The first part of the analysis focuses on predicting the first period grade (G1) using a decision tree regressor. By selecting relevant features, the model's performance is evaluated using the mean squared error (MSE), root mean squared error (RMSE), and the r-squared value as the evaluation metrics. This section aims to provide insights into the early academic performance of students and the initial predictors of their success.

2. Classification Tree of G2T: In the second part, the second period grade (G2) is transformed into a categorical variable (G2T) with five distinct categories: A (17-20), B (13-16), C (9-12), D (5-8),  E (0-4). A decision tree classifier is constructed to categorise these grades and the model's accuracy is assessed. This classification provides a nuanced understanding of the mid-term performance and the categorisation of students' academic levels.

3. Variable Importance Using Random Forest: The final section utilises a Random Forest model to rank the importance of various features in predicting the final grade (G3). By analysing feature importance scores, this section identifies the key determinants of students' final academic outcomes in Mathematics. Understanding these determinants is crucial for developing targeted interventions to support students' academic growth.

This comprehensive analysis not only demonstrates the application of decision tree techniques in educational data mining but also highlights the complex interplay of factors affecting student performance. The insights derived from this study can inform educational strategies and policies aimed at improving student outcomes. By identifying the most influential factors, educators can tailor their approaches to address the specific needs of students, thereby fostering a more conducive learning environment. Through this detailed exploration, the report aims to contribute to a better understanding of student performance dynamics and offer practical insights for enhancing educational practices.

RESULTS AND DISCUSSIONS

Regression Tree for G1

For the purpose of the first objective, a regression tree model is trained. The purpose of predicting G1 is to know whether a student will perform well or not in the first period grade before they take the first period exams, meaning without the information of the second period grades and final grades, hence the variables G2 and G3 are excluded from the model training process.

First, the parameters for the regression tree are decided by performing a grid search to identify the best parameters for the model. Then the model is trained based on the parameters provided by the grid search method. The model evaluation parameters are referred where the MSE which is 7.459, RMSE which is 2.731 and the r-squared value which is 0.24. Based on those evaluation parameters, the model seems to be performing poorly as the MSE and RMSE is high considering the range of the grade is between 0 until 20. The r-squared value also tells that the model is only able to explain 24% of G1 which may mean that the variables used for the model are not enough to predict G1 and other information needs to be collected to achieve a better model. Despite the poor performance of the model, the output, as shown in Figure 1, is explained below.

![3BAE4D00-E9C9-481C-9A3C-7A6D3C6C8A01_4_5005_c](https://github.com/user-attachments/assets/aab60b85-9947-4b81-b562-bcae8af12c54)

Figure 1: Regression Tree

The regression tree results in a tree that provides predictor variables with conditions. The results move by level where if the condition in the level is met, it will go to the left node of the next level and if the conditions are not met, it will move to the right node on the next level instead. Based on Figure 1, the tree created has two levels. The root node is decided by the ‘failure’ variable. This root node, also known as level 0 of the tree, is what splits the data into two which diverges into a group being predicted by ‘absences’ and another by ‘schoolsup_yes’ which are the predictors on the first level of the tree. If the student has more than 0.5 failures, it will move to the right node of the first level which is the number of absences from school, meanwhile for students with 0.5 or less failures, the left node of the first level is the next predictor for their G1 grade which is whether the student receives education support or not. 

On the first level of the tree, the right node predictor is ‘absences’ and the left node is the ‘schoolsup_yes’. The level below is the final level of the tree and contains the leaves of the tree which are the results or the predictions. The right node (‘absences’) is explained first followed by the left node (‘schoolsup_yes’). For the right node ‘absences’, the condition states that the student has to have a number of absences of 1.0 or less to move to the left leaf (7.609 average) or the right leaf (9.634 average) will be the result if the number of absences is more than 1.0. For the left node ‘schoolsup_yes’, the left leaf (11.732 average) will be the result if the student meets the condition of getting 0.5 or less education support and the right leaf (9.250 average) will be the result if the student gets more than 0.5 education support.

To demonstrate, assume the new data given provides information of the three predictors used in the regression tree where a student has 0.7 failures, no absences and no education support. This student does not meet the condition of the root node (failures <= 0.5) hence it moves right for the first level which has a condition of absences must be 1.0 or less. On the first level, the student meets the conditions (absences <= 1.0) and moves to the left leaf on the next level which results in an average G1 of 7.609. This means that students that meet the conditions to get to this particular leaf have an average grade of 7.609.

To conclude for the regression tree, the best number of levels, as decided by the grid search, is a tree with two levels. The best variables for a regression tree with two levels using this dataset are the ‘failures’, ‘absences’ and ‘schoolsup’ variables. However, this model is not the best due to the high MSE and RMSE and low r-squared value. It may be further improved with the addition of other variables.


Classification Tree with G2T

![33EDA769-DDDE-4C92-BE13-FEB170AD013C_4_5005_c](https://github.com/user-attachments/assets/44700ae1-9e4b-43bc-bc4a-fe9730d97a18)

Figure 2: Classification Tree

In our analysis, we aimed to categorise students' second period grades (G2) into a newly defined variable (G2T) with five categories such A(scoring from 20-17), B(scoring from 16-13), C(scoring from 12-9), D(scoring from 8-5) and E(scoring from 4-0). We utilised Classification Tree to predict G2T, excluding the variables G2, G2T and G3 from the analysis. Additionally, we perform Cross Validation (CV) method to select the best parameters using function GridSearchCV from sklearn package.

This method will give us the ideal parameters for Classification Tree and ultimately provide valuable insights into the students’ performance. The decision tree revealed that the first period grade (G1) is the most significant predictor of G2T, with an initial split threshold at a score of 12.5 while Gini impurity value of 0.706 with 316 samples. The samples value distribution of 18, 92, 127, 65 and 14 represent the targeted classes (G2T categories) within this node. This split suggests that G1 values less than or equal to 12.5 are directed to the left branch, while values greater than 12.5 are directed to the right branch. 

The left branch further splits at G1 scores less than or equal to 8.5, representing a mixed class distribution with a moderate impurity level of 0.615 with 213 samples having value distribution of 0, 22, 112, 65 and 14. It shows that students with G1 scores less than or equal to 8.5 are spread across different outcome categories with a notable concentration in specific classes.

On the other hand, in the right branch from the root, the first split occurs when G1 is less than or equal to 15.5 with a Gini index of 0.486. This shows moderate impurity among the 103 samples and a distribution of 18, 70, 15, 0 and 0. This node further splits into two sub-branches whereby the left node has a Gini index of 0.327, reflecting lower impurity among the 68 samples with the distribution of 0, 54, 14, 0 and 0.  This indicates a majority of class B. Meanwhile the right leaf shows a Gini index of 0.526, indicating moderate impurity among the 35 samples with distribution of 18, 16, 1, 0 and 0. This shows a majority of classes A and B, respectively.

The feature G1 emerges as a significant predictor for classifying G2T, with multiple splits based on its value. Lower G1 values (<= 8.5) are associated with higher frequencies of classes C and D, while intermediate G1 values (> 12.5 and <= 15.5) show a higher frequency of classes B and C, and higher G1 values (> 15.5) are associated with classes A and B. The impurity, measured by the Gini index, decreases as one moves down the tree, indicating better classification purity in the leaf nodes. For instance, the left most leaf node (G1 <= 8.5) has a Gini index of 0.533, while the right most leaf node (G1 > 15.5) has a Gini index of 0.526. The decision tree thus provides a structured approach for predicting the G2T categories based on G1, demonstrating how various G1 thresholds influence academic performance categories in the mathematics subject. This model offers valuable insights into the role of G1 in determining student performance categories, suggesting its potential utility for educational data analysis.

Random Forest

In question 3, the Random Forest model was utilised to estimate and rank the variables according to their importance in predicting the students’ final grade (G3) in mathematics. The influence of each variable is measured by how much it contributes to reducing the prediction error (MSE) in the model. The Random Forest model was trained with hyperparameter tuning approach using Grid Search Cross Validation. 


Table 1: The best hyperparameter 
Max_depth: 10
Max_features: sqrt
Min_samples_leaf: 1
Min_samples_split: 2
n_estimators: 200
random_state: 1

The model is then feeded the training and testing datasets and the results are shown below:

![C73C606D-CE7C-4B10-B3E0-1AF0ED3AA151_4_5005_c](https://github.com/user-attachments/assets/9935fd9c-3062-452a-919c-cf1e80fc2b05)

Figure 3: Training and testing performance

Based on the results, the model demonstrates strong performance on the training set, with an R² score of 0.9655, which indicates that the model explains 96.55% of the variance in the student's final grade (G3). The testing set performance showed a slight decrease in the R² score of 0.7978, but it still indicates a good level of explanation of approximately 79.78% of the variance in G3.

![9037B53A-A087-4DD1-9772-2FB7D970B4C9](https://github.com/user-attachments/assets/7157d833-2fa6-4f8c-bba4-98256449d801)

Figure 4: Top 10 Most Important Features

The barplot above visualises the ‘Top 10 Most Important Features’ in predicting the students' final grade (G3) in Mathematics according to the Random Forest model results. The X-axis represents the importance score of each feature contributing to the prediction of the final grade, while the Y-axis lists the top 10 features according to their order of importance, with the top being the most important. 

The number 1 most significant predictor of G3 is the second period grade (G2) with a score of 0.354276. This suggests that students who performed well in the second period will likely perform well in the final period due to consistency of studying and understanding the lessons or materials. Typically, the progression from G2 to G3 is reflected in the students' ability to maintain academic performance. The following feature with a 0.216034 importance score is a first-period grade (G1). Understandably the performance of the first period usually sets the foundation for subsequent academic success, which indicates the students’ grasp of the subject matter. 

The number of absences also impacts student performance with a score of 0.071744. Attendance is crucial for continuous learning and understanding. Recurring absence will lead to gaps in learning and knowledge, hence leading to lower grades. Failures feature the 4th in importance score with 0.034973 which indicates the number of past class failures. Typically students with a history in failing subjects may struggle more in their current studies. This is possibly due to underlying academic challenges or personal matters that might overwhelm the student. Addressing or involvement targeted at students with past failed subjects could possibly help in providing support to improve their academic results. 

The age of a student too can affect the grades of a student with an importance score of 0.021594. The difference in age-related maturity and intellectual development among students in the same class can lead to diversity in academic performance. Students with a more active social life also affect the students grade, for example, going out with friends (goout) has an importance score of 0.018802. Having a balanced social life and study is crucial, an excessive amount of socialising can stray a student from study time.  Parental education is also important for student performance albeit mother and father education has a sheer difference. Based on the importance score there is a slight difference between mothers education (medu) with 0.018513 and fathers education (fedu) with 0.016539. Having a more supportive home environment, where education is emphasised by parents and is linked to better academic performance. 

Academic performance is strongly linked to study time of students on their studies. In the results study time has an importance score of 0.015944. This is due to the amount of time dedicated to studying which can have a positive impact on their grades. By reinforcing their learning, reviewing materials, practicing applications in their own time at their own pace, students are able to enhance their understanding and academic outcomes. Lastly, the health of a student is a significant factor that can impact a student's performance, with an importance score of 0.015914. Healthy students are more likely to attend classes. For optimal brain function, students need proper nutrition and sleep, which helps with their concentration in class. Mental health conditions like stress, anxiety or depression can severely impact students' performance.  Poor mental health can also lead to behavioral issues. Physical, mental and emotional well-being are connected with cognitive functioning, attendance, behaviour and overall ability to succeed academically. 
Overall, the analysis revealed the most influential variables in predicting a student’s final grade in Mathematics are G2, G1, attendance and past failures. Parental education and lifestyle choices are also the top factors that play important roles in a student’s ability to perform.

CONCLUSIONS

In conclusion, three different models were trained for three different purposes. The first period grades (G1) was predicted using a regression tree, the second period grades (G2) was predicted using a classification tree and the final grades (G3) was predicted using a random forest model. Based on what has been presented in the section above, the regression tree model is not the best for predicting the G1 grades and it may be further improved if other data that may explain the grades of students is used to train the model. Next, the classification tree presents a better tree model than the first model. The difference between these two models is the existence of G1 data as the predictor variable in the classification tree model while the regression tree model only relied on external factors as its predictors. This shows past performance data played a crucial role in predicting the performance of a student. 

Finally, the random forest model has a decent r-squared value which indicates that it can explain a decent portion of the target variable (G3). The best predictors for the random forest model are G2 and G1 followed by absences and failures. Although the resulting models are the best based on the provided dataset, these models still lack in terms of fit or accuracy. More data that can possibly tell more about a student’s academic achievements is needed to refine these models as the readily available data do not seem to give these models enough relevant data for predicting G1, G2 and G3.
