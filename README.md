<h1>Bank Customer Attrition Prediction and Feature Importance</h1>

<p>This project focuses on analyzing the root causes and factors leading to customer attrition in a Bank. The aim is to build a predictive model to identify customers at risk of churning, enabling proactive measures to prevent it.</p>

<a href="https://github.com/jvontama96/AirlinesCustomerClustering_RFM_KMeans/blob/main/Customer_Airlines_Clustering_KMeans_RFM.ipynb">Full Project Documentation</a></p>

<h2>Data Processing</h2>
<ul>
  <li>Dataset: <a href="https://raw.githubusercontent.com/hadimaster65555/dataset_for_teaching/main/dataset/bank_churn_dataset/bank_churn_data.csv">Bank Churn Dataset</a></li>
  <li>Data Details:</li>
</ul>

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10127 entries, 0 to 10126
Data columns (total 21 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   user_id                   10127 non-null  int64  
 1   attrition_flag            10127 non-null  object 
 2   customer_age              10127 non-null  int64  
 3   gender                    10127 non-null  object 
 4   dependent_count           10127 non-null  int64  
 5   education_level           10127 non-null  object 
 6   marital_status            10127 non-null  object 
 7   income_category           10127 non-null  object 
 8   card_category             10127 non-null  object 
 9   months_on_book            10127 non-null  int64  
 10  total_relationship_count  10127 non-null  int64  
 11  months_inactive_12_mon    10127 non-null  int64  
 12  contacts_count_12_mon     10127 non-null  int64  
 13  credit_limit              10127 non-null  float64
 14  total_revolving_bal       10127 non-null  int64  
 15  avg_open_to_buy           10127 non-null  float64
 16  total_amt_chng_q4_q1      10127 non-null  float64
 17  total_trans_amt           10127 non-null  int64  
 18  total_trans_ct            10127 non-null  int64  
 19  total_ct_chng_q4_q1       10127 non-null  float64
 20  avg_utilization_ratio     10127 non-null  float64
dtypes: float64(5), int64(10), object(6)
memory usage: 1.6+ MB
</pre>

<h2>Exploratory Data Analysis (EDA)</h2>
<ul>
  <li>1. Analysis of Customer Age Influence on Churn</li>
   <img 
     src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/Data Analysis/Analysis on age correlation to churn/Distribution of customer age.png?raw=true" alt="age1" style="width:60%; max-width:450px;">
  <p>The distribution of customer ages shows that most users are centered around 40-50 years old, which is the millennial generation.</p>
  <img 
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/Data Analysis/Analysis on age correlation to churn/Customer top 10 age.png?raw=true" alt="age2" style="width:60%; max-width:450px;">
  <p>if we more specific with the analysis to observe the top 10 age with the most user we can find it around age of 40 - 51 with the age of 44 with the most user.</p>
  <img 
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/Data Analysis/Analysis on age correlation to churn/top 10 age with the highest churn rate.png?raw=true" alt="age3" style="width:60%; max-width:450px;">
  <p>The analysis of the "churn rate" shows the percentage ratio of attrited customers within the total customers of each age group. By observing the age group with the highest churn possibility, we look at the percentage of attrited customers. If we only consider the count of total attrited customers, the result will show ages 40-50 as having the highest churn due to this group having the most customers. However, this would not accurately reflect the churn trend based on age.</p>

<p>The plot shows that ages 66 and 68 have the highest percentage of attrited customers within their age group, indicating that these ages have the highest possibility of churn. Additionally, the trend shows that the older the age, the higher the percentage of attrited customers. This suggests that older users might have less need for online financing platforms, or that the platform's features/UI are too complicated for older users to use.</p>
  <li>2. Education level and income category influence towards churn</li>
   <img 
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/Data Analysis/Education level and income category influence towards churn/income category distribution in education level.png?raw=true" alt="age3" style="width:60%; max-width:450px;">
  <p>- A more detailed distribution analysis of the income category for each education level shows that there is no significant impact of education level on the income category of the users. The income category "Less than $40K" has the highest percentage across all education level groups.</p>

 <p>- Logically, higher education should correlate with higher income. However, the distribution analysis shows the opposite: while the highest income category "$120K+" has the highest percentage within the doctorate education level, the second highest percentage is within the uneducated group.</p>

 <p>- We can conclude that most users of the financing platform fall into the income category "Less than $40K," regardless of their educational background. This may indicate that one of the main motivations for using these platforms is financial necessity among lower-income individuals.</p>
  <img 
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/Data Analysis/Education level and income category influence towards churn/churn rate by education.png?raw=true" alt="age3" style="width:60%; max-width:450px;">
  <img 
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/Data Analysis/Education level and income category influence towards churn/Churn rate by income category.png?raw=true" alt="age3" style="width:60%; max-width:450px;">
<p>Lastly, we break down the analysis to observe the relationship between income category and education level with the percentage of attrited customers within each group.</p>

<p>The analysis shows that the <b>Doctorate</b> education level and the <b>$120K+</b> income category have the highest percentage of attrited customers within their groups. This indicates that users with a Doctorate education level or users with an income of $120K+ have a higher possibility of churning.</p>

<p>However, the overall graphic shows no significant relationship between higher income and the likelihood of a user stopping the use of the platform. This is evident because the <b>Less than $40K</b> income category also has a high percentage of churned customers. While the higher education levels (<b>Doctorate</b> and <b>Post-graduate</b>) have a higher percentage of churned customers, indicating that users with high education levels (beyond the graduate level) have a higher possibility of churning.</p>

  <li>3. Gender category influence towards churn</li>
  <img 
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/Data Analysis/Gender category influence towards churn/Churn Rate by Gender.png?raw=true" alt="age3" style="width:60%; max-width:450px;">
  <p>The graph shows that the female gender has a higher percentage of churned users, although this difference is not statistically significant compared to the male group.</p> 
<img 
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/Data Analysis/Gender category influence towards churn/Chi Square test.JPG?raw=true" alt="age3" style="width:60%; max-width:450px;">
  <p>To further refine our analysis, we will use the Chi-square test to statistically analyze if there is any significant difference between males and females in terms of churn likelihood.</p>

<p>the results show:<p>

<p>- Significance Level: The p-value is much lower than the common significance level of 0.05.</p>

<p>- Rejecting the Null Hypothesis: This low p-value indicates that we can reject the null hypothesis, which states that there is no relationship between gender and churn likelihood.</p>

<p>- Conclusion: There is a statistically significant relationship between gender and churn likelihood. This suggests that the likelihood of customer churn is different between genders.<p>

  <li>4. Correlation between length of being a customer and churn likelihood</li>
  <img 
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/Data Analysis/Correlation between length of being a customer and churn likelihood/distribution of length.png?raw=true" alt="age3" style="width:60%; max-width:450px;">
  <p>To understand the relationship between the length of time as a customer (months_on_book) and the possibility of churn, we will analyze the distribution based on months_on_book, which represents the period of being a customer in months.</p>

<p>First, based on the distribution of users by months_on_book, it shows that most users stay as customers for around 35 to 40 months.</p>
  <img
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/Data Analysis/Correlation between length of being a customer and churn likelihood/percentage of attrited customers by months on book.png?raw=true" alt="age3" style="width:60%; max-width:450px;">
  <p>Next, we will analyze the percentage of churned customers within each month that the user has been a customer to observe any relationship.
We can see that most customers possibly churn in the beginning months of using the platform. Additionally, after using the platform for a long time, there are peaks at around 15 months and 50 months where there is a higher percentage of churned customers.

It can be concluded that users may use the platform for either short-term financing or long-term financing, based on these patterns.</p>
  <li>5. Impact of interactions with the bank in the last year on churn likelihood</li>
   <img
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/Data Analysis/Impact of interactions with the bank in the last year on churn likelihood/distribution number of interactions with bank in last 12 months.png?raw=true" alt="age3" style="width:60%; max-width:450px;">
  <p>To understand the impact of interactions with the bank in the last year on churn likelihood, we will analyze contacts_count_12_mon, which represents the number of interactions between the bank and the customer in the last 12 months.
    The graph shows that most customers interacted three times within the last 12 months.</p>
   <img
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/Data Analysis/Impact of interactions with the bank in the last year on churn likelihood/percentage of attrited customer by total interaction.png?raw=true" alt="age3" style="width:60%; max-width:450px;">
  <p>The graph shows that customers who had 6 interactions with the bank are all churned customers (100% churn rate). Overall, the trend in the graph indicates that higher interaction frequency in the last 12 months correlates with a higher percentage of churned customers.</p>
</ul>

<h2>Modeling Process</h2>
<ul>
<h3>Data Processing <a href="https://github.com/jvontama96/AirlinesCustomerClustering_RFM_KMeans/blob/main/Customer_Airlines_Clustering_KMeans_RFM.ipynb">(Check Full Project Documentation)</a></p></h3>
  <ul>
  <li>Encoding</li>
  <li>Data Splitting (<strong>Target Feature: 'Attrition Flag', 0: Churn, 1: not churn</strong>)</li>
  <li>Feature Selection</li>
  <li>Outlier Handling</li>
  </ul>
<h3>Model Training</h3>
   <ul>
  <li>Interpretable Models: Decision Tree and KNN</li>
      <img
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/ModelEvaluation_and_FeatureImportance/DT_training.JPG?raw=true" alt="age3" style="width:60%; max-width:450px;">
       <img
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/ModelEvaluation_and_FeatureImportance/KNN_training.JPG?raw=true" alt="age3" style="width:60%; max-width:450px;">
  <li>Non-Interpretable Models: Random Forest</li>
      <img
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/ModelEvaluation_and_FeatureImportance/RF_training.JPG?raw=true" alt="age3" style="width:60%; max-width:450px;">
  </ul>
  <h3>Model Evaluation</h3>
  <ul>
    <li>Confusion Matrix</li>
    <ul>
    <li>1. Confusion Matrix Decision Tree</li>
    <img
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/ModelEvaluation_and_FeatureImportance/Confusion_DecisionTree.png?raw=true" alt="age3" style="width:60%; max-width:450px;">
    <li>2. Confusion Matrix KNN</li>
        <img
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/ModelEvaluation_and_FeatureImportance/Confusion_KNN.png?raw=true" alt="age3" style="width:60%; max-width:450px;">
      <li>3. Confusion Matrix Random Forest</li>
        <img
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/ModelEvaluation_and_FeatureImportance/Confusion_RandomForest.png?raw=true" alt="age3" style="width:60%; max-width:450px;">
      </ul>
    <li>Classification Report</li>
      <img
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/ModelEvaluation_and_FeatureImportance/Model_evaluation_AttritionCust.JPG?raw=true" alt="age3" style="width:60%; max-width:450px;">
<h3>Best Model - Random Forest:</h3>
<ul>
  <li>
    <strong>Highest Precision (0.85):</strong> The Random Forest model has the highest precision for predicting churn. This means that 85% of the customers it predicts as likely to churn do indeed churn. This high precision reduces the risk of false positives (i.e., incorrectly predicting that a customer will churn).
  </li>
  <li>
    <strong>Trade-off with Recall (0.46):</strong> While the recall is lower, meaning it misses some customers who do churn, our primary focus is precision. This trade-off is acceptable given our objective to minimize false alarms.
  </li>
</ul>
<p>
  <strong>Conclusion:</strong> Using the Random Forest model, we can more accurately identify customers at risk of churning. This allows us to target retention efforts more effectively and reduce unnecessary interventions for customers who are not at risk.
</p>
    
<li>Model ROC Curves Analysis</li>
 <ul>
    <li>1. ROC Decision Tree</li>
 <img
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/ModelEvaluation_and_FeatureImportance/ROC_DT.png?raw=true" alt="age3" style="width:60%; max-width:450px;"> 
    <li>2. ROC KNN</li>
    <img
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/ModelEvaluation_and_FeatureImportance/ROC_KNN.png?raw=true" alt="age3" style="width:60%; max-width:450px;">
   <li>3. ROC Random Forest</li>
    <img
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/ModelEvaluation_and_FeatureImportance/ROC_RF.png?raw=true" alt="age3" style="width:60%; max-width:450px;">
   <p>Based on all the model ROC Curves, we can identify that all the ROC curve lines (represented in blue) are above the random classifier line (black dotted line). This means that all the models are better at predicting class 0 (churn class customer) compared to predicting it without a model (only by reading data trends, relationships with certain features, etc.).</p>
 </ul>
 </ul>
 
<h2>Feature Importance: Model Agnostic Methods</h2>
  <p>The best model is Random Forest. Because Random Forest is not an inherently interpretable model, we will focus on analyzing feature importance using Model Agnostic Methods.</p>
  <ul>
  <h3>1. Drop-out Loss</h3>
       <img
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/ModelEvaluation_and_FeatureImportance/Dropout_Loss.png?raw=true" alt="age3" style="width:100%; max-width:600px;">
    <h3>Insights from the Feature Importance (Dropout Loss Plot):</h3>
<h4>Key Features:</h4>
<ul>
  <li><strong>Total Transaction Amount (total_trans_amt):</strong> The most critical feature influencing customer attrition, with the highest drop-out loss (+0.094). This suggests that customers with low transaction amounts are more likely to churn.</li>
  <li><strong>Total Revolving Balance (total_revolving_bal):</strong> The second most important feature (+0.061), indicating that customers with high revolving balances may exhibit attrition tendencies.</li>
  <li><strong>Change in Total Transactions (total_ct_chng_q4_q1):</strong> An important feature (+0.042), highlighting that a decline in transaction activity between quarters signals potential churn.</li>
  <li><strong>Total Relationship Count (total_relationship_count):</strong> Customers with fewer products or accounts tied to the bank (+0.034) are at a higher risk of attrition.</li>
</ul>
<h4>Supporting Features:</h4>
<ul>
  <li><strong>Change in Total Amount (total_amt_chng_q4_q1):</strong> Indicates variations in spending habits, possibly hinting at disengagement.</li>
  <li><strong>Contact Frequency (contacts_count_12_mon):</strong> Customers contacted frequently may be high-risk or less engaged.</li>
  <li><strong>Credit Limit and Months of Inactivity:</strong> Lower credit limits and extended inactivity periods correlate with potential churn but have a smaller impact.</li>
</ul>
<h3>Business Recommendations to Prevent Churn:</h3>
<ul>
<li>Boost Engagement with Low-Transaction Customers:

- Introduce transaction-based rewards programs (e.g., cashback or discounts) to encourage higher spending.
- Provide personalized financial tools to guide customers on maximizing account benefits.</li>

<li>Manage High Revolving Balances:

- Offer interest rate reductions or balance consolidation plans to alleviate financial pressure.
- Launch credit management webinars to educate customers on sustainable usage.</li>

<li>Proactively Address Declining Activity:

- Implement early-warning systems to detect drops in transactions and initiate targeted re-engagement campaigns.
- Provide time-limited offers (e.g., discounts or loyalty points) to stimulate renewed activity.</li>

<li>Enhance Customer Relationships:

- Encourage adoption of additional products (e.g., savings accounts, loans) through personalized cross-selling.
- Assign dedicated account managers to high-risk customers to build trust and engagement.</li>

<li>Optimize Communication Strategies:

- Use data-driven insights to tailor outreach frequency and content, ensuring relevance and avoiding over-contacting.
- Leverage customer feedback to align services with their needs, improving satisfaction and loyalty.</li>
</ul>

 <h3>2. Partial Dependence Plot</h3>
  <img
  src="https://github.com/jvontama96/BankCustomerAttritionPrediction-and-FeatureImportance/blob/main/ModelEvaluation_and_FeatureImportance/Partial_ Dependence_Plot.png?raw=true" alt="age3" style="width:100%; max-width:600px;">
  <ul>
<h3>Insights from Partial Dependence Plot (Features Impacting Churn Prediction):</h3>
<h4>Key Features:</h4>
<ul>
  <li><strong>Total Transaction Amount (total_trans_amt):</strong> Churn probability significantly decreases as the total transaction amount increases. Customers with lower spending levels are more likely to churn.</li>
  <li><strong>Total Revolving Balance (total_revolving_bal):</strong> Customers with high revolving balances exhibit a higher likelihood of churn, indicating financial stress or dissatisfaction with credit-related features.</li>
  <li><strong>Total Relationship Count (total_relationship_count):</strong> Customers with fewer relationships (fewer products or services) tied to the bank are more likely to churn.</li>
  <li><strong>Months of Inactivity (months_inactive_12_mon):</strong> Churn probability increases with a higher number of inactive months.</li>
  <li><strong>Change in Total Transactions (total_ct_chng_q4_q1):</strong> A decline in transaction activity between quarters correlates with a higher churn probability.</li>
</ul>
<h3> Business Recommendations</h3>
<h4>Personalized Engagement Strategies</h4>
<ul>
    <li><strong>Customer Segmentation:</strong> Segment customers based on key churn indicators such as low transactions or high inactivity.</li>
    <li><strong>Tailored Communication Plans:</strong> Develop customized communication strategies to re-engage at-risk customers, emphasizing personalized rewards and offers.</li>
</ul>

<h4>Proactive Monitoring and Alerts</h4>
<ul>
    <li><strong>Predictive Analysis:</strong> Use predictive models to identify customers with declining activity or high financial stress.</li>
    <li><strong>Real-time Alerts:</strong> Send proactive notifications about available offers, financial tips, or product recommendations.</li>
</ul>

<h4>Product Diversification and Bundling</h4>
<ul>
    <li><strong>Cross-Selling and Up-Selling:</strong> Encourage customers to explore additional banking services by offering discounts or rewards for multi-product usage.</li>
    <li><strong>Bundle Offers:</strong> Focus on building deeper customer relationships by promoting bundled financial solutions.</li>
</ul>

<h4>Targeted Financial Support</h4>
<ul>
    <li><strong>Personalized Financial Assistance:</strong> Offer customized credit solutions or consultations for customers with high revolving balances or financial stress.</li>
    <li><strong>Financial Health Programs:</strong> Build loyalty by showing concern for customers' financial well-being through programs or features focused on improving their financial health.</li>
</ul>

<h4>Incentives for Reactivation and Retention</h4>
<ul>
    <li><strong>Exclusive Incentives:</strong> Offer exclusive incentives for customers who re-engage after periods of inactivity.</li>
    <li><strong>Loyalty Programs:</strong> Launch loyalty programs tied to transaction volume to ensure ongoing engagement and customer satisfaction.</li>
</ul>
    </ul>
  </ul>
</ul>
