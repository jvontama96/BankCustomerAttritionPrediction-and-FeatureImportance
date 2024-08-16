<h1>Bank Customer Attrition Prediction and Feature Importance</h1>

<p>This project focuses on analyzing the root causes and factors leading to customer attrition in a Bank. The aim is to build a predictive model to identify customers at risk of churning, enabling proactive measures to prevent it.</p>

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
  <li>Distribution of customer age and top 10 ages with the highest churn rate</li>
  <li>Education level and income category influence towards churn</li>
  <li>Gender category influence towards churn</li>
  <li>Correlation between length of being a customer and churn likelihood</li>
  <li>Impact of interactions with the bank in the last year on churn likelihood</li>
</ul>

<h2>Modeling and Analysis</h2>
<ul>
  <li>Encoding</li>
  <li>Data Splitting</li>
  <li>Correlinality Feature: Drop high colinearity</li>
  <li>Outlier Handling</li>
  <li>Interpretable Models: Logistic Regression and KNN</li>
  <li>GridSearch CV Tuning</li>
  <li>Non-Interpretable Models: Random Forest</li>
  <li>GridSearch CV Tuning</li>
  <li>Model Evaluation</li>
  <ul>
    <li>Confusion Matrix</li>
    <li>Classification Report</li>
  </ul>
</ul>

<h3>Best Model - Random Forest:</h3>
<ul>
  <li><strong>Highest Precision (0.85):</strong> The Random Forest model has the highest precision for predicting churn. This means that 85% of the customers it predicts as likely to churn do indeed churn. This high precision reduces the risk of false positives (i.e., incorrectly predicting that a customer will churn).</li>
  <li><strong>Trade-off with Recall (0.46):</strong> While the recall is lower, meaning it misses some customers who do churn, our primary focus is precision. This trade-off is acceptable given our objective to minimize false alarms.</li>
</ul>

<h3>Conclusion:</h3>
<p>Using the Random Forest model, we can more accurately identify customers at risk of churning. This allows us to target retention efforts more effectively and reduce unnecessary interventions for customers who are not at risk.</p>

<h3>Additional Analysis</h3>
<ul>
  <li>ROC Curves</li>
  <li>Feature Importance: Model Agnostic Methods</li>
  <li>Dependence Plot:</li>
  <ul>
    <li>Based on the dependence plot observation, we can see the impact of the <code>contacts_count_12_months</code> feature. It indicates that customers who interact 2-3 times within the last 12 months are more likely to churn. Additionally, customers who have more than 3 interactions are even more likely to churn.</li>
    <li>We can recommend several actions based on this insight:</li>
    <ul>
      <li>Observe the products and features on the online platform that most customers use in 2-3 interactions. Evaluate the UX and optimize features to make the platform more user-friendly, especially for older or less tech-savvy customers.</li>
      <li>Utilize the interactions to provide proactive customer support and assistance. Identify pain points or issues that customers may face and address them promptly to enhance customer satisfaction.</li>
      <li>Use insights from customer interactions to offer personalized incentives or rewards. This could include targeted discounts on services, exclusive offers for loyal customers, or benefits that align with their tier of financial needs.</li>
    </ul>
  </ul>
</ul>
