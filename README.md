# ghr_Telecom_Churn_Logistic_Regression_DT_RF
To reduce customer churn, telecom companies need to predict which customers are at high risk of churn. In this project, we will analyse customer-level data of a leading telecom firm, build predictive models to identify customers at high risk of churn and identify the main indicators of churn.

## Table of Contents
  * [What: Project Overview](#what-project-overview)
  * [Why: Motivation](#why-motivation)
  * [How: Process Involved](#how-process-involved)
  * [Project Problem Statement Description](#project-problem-statement)
  * [Directory Tree](#directory-tree)
  

## WHAT: Project Overview 
### Business problem overview
In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition.

For many incumbent operators, retaining high profitable customers is the number one business goal.

To reduce customer churn, telecom companies need to predict which customers are at high risk of churn.In this project, you will analyse customer-level data of a leading telecom firm, build predictive models to identify customers at high risk of churn and identify the main indicators of churn.

## WHY: Motivation
This project was part of Upgrads PGDDS(Post Graduate Program in Data Science) with IIITB University Bangalore

## HOW: Process Involved
- This section mentions EDA Process, ML algorithms, Libraries, stragtegy used if any etc
- I used simple Python programming knowledge with numpy, pandas and lambda functions.
- I used data visualization libraries: `seaborn` and `matpotlib` for this project.
- After performing EDA I have got a hang of which features are important and go into the model to explain it better w.r.t to **target variable `converted`(Lead is converted or not, just two classes here).**


Process Followed: 
- First I inspected the Data Dictionary and tried to make
sense(intuitively) of the variables that could go into the model.
-  Visualized or comprehended the EDA process that should go into the model. The main
challenge faced was to reduce the levels of categorical variables so that we have less
number of dummy variables. This was done by using a data dictionary for respective
categorical fields. This was time consuming.
-  Also deleted /dropped the least important category while creating dummies for
categorical variables. Eg “Others” was the category which was dropped explicitly rather
than using “drop_first=True”.
-  Another Learning is we just cannot strictly follow the low p_value and High VIF value to
be dropped. Along this process we also need to have business sense as to which
variable needs to be dropped and what information along with that variable is getting
dropped from the model. The best business sense of variables to be included can be got
by drawing a heat map of all the features/variables against the Target variable. This
gives solid intuition while entering the feature selected process.
- Also the model should be open to **scale and change** when business requirements
change, we should develop the model keeping this in mind.
-  Based on business understanding we should either work towards increasing sensitivity
or precision based on business requirements. **In Our case it was increasing sensitivity while not letting go of Specificity by
large values.**

## Project Problem Statement

### Problem Statement
<!--![lead_funnel_image](./presentations/images/lead_funnel_image.jpg?raw=true "Leads Funnel") -->
<!-- ![lead_funnel_image](https://user-images.githubusercontent.com/14209223/149206507-62bf586c-e02b-41fc-9674-44aa3bb8931c.jpg) <!-- I simply dragged and dropped the image-->

### Understanding and defining churn
There are two main models of payment in the telecom industry - postpaid (customers pay a monthly/annual bill after using the services) and prepaid (customers pay/recharge with a certain amount in advance and then use the services).In the postpaid model, when customers want to switch to another operator, they usually inform the existing operator to terminate the services, and you directly know that this is an instance of churn.

However, in the prepaid model, customers who want to switch to another network can simply stop using the services without any notice, and it is hard to know whether someone has actually churned or is simply not using the services temporarily (e.g. someone may be on a trip abroad for a month or two and then intend to resume using the services again).

Thus, churn prediction is usually more critical (and non-trivial) for prepaid customers, and the term ‘churn’ should be defined carefully.  Also, prepaid is the most common model in India and Southeast Asia, while postpaid is more common in Europe in North America.

This project is based on the **Indian and Southeast Asian market.**

### Definitions of churn
There are various ways to define churn, such as:

- **Revenue-based churn:** Customers who have not utilised any revenue-generating facilities such as mobile internet, outgoing calls, SMS etc. over a given period of time. One could also use aggregate metrics such as ‘customers who have generated less than INR 4 per month in total/average/median revenue’.

The main shortcoming of this definition is that there are customers who only receive calls/SMSes from their wage-earning counterparts, i.e. they don’t generate revenue but use the services. For example, many users in rural areas only receive calls from their wage-earning siblings in urban areas.

- **Usage-based churn:** Customers who have not done any usage, either incoming or outgoing - in terms of calls, internet etc. over a period of time.
A potential shortcoming of this definition is that when the customer has stopped using the services for a while, it may be too late to take any corrective actions to retain them. For e.g., if you define churn based on a ‘two-months zero usage’ period, predicting churn could be useless since by that time the customer would have already switched to another operator.

In this project, you will use the **usage-based definition** to define churn.

### High-value churn
In the Indian and the Southeast Asian market, approximately 80% of revenue comes from the top 20% customers (called high-value customers). Thus, if we can reduce the churn of the high-value customers, we will be able to reduce significant revenue leakage.

In this project, you will define high-value customers based on a certain metric (mentioned later below) and predict churn only on high-value customers.

### Understanding the business objective and the data
The dataset contains customer-level information for a span of four consecutive months - June, July, August and September. The months are encoded as 6, 7, 8 and 9, respectively. 

The business objective is to predict the churn in the last (i.e. the ninth) month using the data (features) from the first three months. To do this task well, understanding the typical customer behaviour during churn will be helpful.

### Understanding customer behaviour during churn
Customers usually do not decide to switch to another competitor instantly, but rather over a period of time (this is especially applicable to high-value customers). In churn prediction, we assume that there are **three phases of customer lifecycle :**

- The ‘good’ phase: In this phase, the customer is happy with the service and behaves as usual.
- The ‘action’ phase: The customer experience starts to sore in this phase, for e.g. he/she gets a compelling offer from a  competitor, faces unjust charges, becomes unhappy with service quality etc. In this phase, the customer usually shows different behaviour than the ‘good’ months. Also, it is crucial to identify high-churn-risk customers in this phase, since some corrective actions can be taken at this point (such as matching the competitor’s offer/improving the service quality etc.)
- The ‘churn’ phase: In this phase, the customer is said to have churned. You define churn based on this phase. Also, it is important to note that at the time of prediction (i.e. the action months), this data is not available to you for prediction. Thus, after tagging churn as 1/0 based on this phase, you discard all data corresponding to this phase.

In this case, since you are working over a four-month window, the first two months are the ‘good’ phase, the third month is the ‘action’ phase, while the fourth month is the ‘churn’ phase.

### Data Dictionary
- This is provided under references folder
- The data dictionary contains meanings of abbreviations. Some frequent ones are loc (local), IC (incoming), OG (outgoing), T2T (telecom operator to telecom operator), T2O (telecom operator to another operator), RECH (recharge) etc.
- The attributes containing 6, 7, 8, 9 as suffixes imply that those correspond to the months 6, 7, 8, 9 respectively.

### Data preparation
The following data preparation steps are crucial for this problem:
1. Filter high-value customers:

As mentioned above, you need to predict churn only for high-value customers. Define high-value customers as follows: Those who have recharged with an amount more than or equal to X, where X is the **70th percentile** of the average recharge amount in the first two months (the good phase).

After filtering the high-value customers, you should get about 30k rows.

2. Tag churners and remove attributes of the churn phase

Now tag the churned customers (churn=1, else 0) based on the fourth month as follows: Those who have not made any calls (either incoming or outgoing) AND have not used mobile internet even once in the churn phase. The attributes you need to use to tag churners are:
- total_ic_mou_9
- total_og_mou_9
- vol_2g_mb_9
- vol_3g_mb_9

After tagging churners, remove all the attributes corresponding to the churn phase (all attributes having ‘ _9’, etc. in their names).

### Modelling
Build models to predict churn. The predictive model that you’re going to build will serve two purposes:

- It will be used to predict whether a high-value customer will churn or not, in near future (i.e. churn phase). By knowing this, the company can take action steps such as providing special plans, discounts on recharge etc.
- It will be used to identify important variables that are strong predictors of churn. These variables may also indicate why customers choose to switch to other networks.

In some cases, both of the above-stated goals can be achieved by a single machine learning model. Also, since the rate of churn is typically low (about 5-10%, this is called class-imbalance) - try using techniques to handle class imbalance. 

**You can take the following suggestive steps to build the model:**
- Preprocess data (convert columns to appropriate formats, handle missing values, etc.)
- Conduct appropriate exploratory analysis to extract useful insights (whether directly useful for business or for eventual modelling/feature engineering).
- Derive new features.
- Train a variety of models, tune model hyperparameters, etc. (handle class imbalance using appropriate techniques).
- Evaluate the models using appropriate evaluation metrics. Note that it is more important to identify churners than the non-churners accurately - choose an appropriate   evaluation metric which reflects this business goal.
- Finally, choose a model based on some evaluation metric.

Therefore, build another model with the main objective of identifying important predictor attributes which help the business understand indicators of churn. A good choice to identify important variables is a **logistic regression model or a model from the tree family.** In the case of logistic regression, make sure to handle multicollinearity.

After identifying important predictors, display them visually - you can use plots, summary tables etc. - whatever you think best conveys the importance of features.

Finally, recommend strategies to manage customer churn based on your observations.

**Note:** Everything has to be submitted in one Jupyter notebook.

<!--## Directory Tree 
```
├── app 
│   ├── __init__.py
│   ├── main.py
│   ├── model
│   ├── static
│   └── templates
├── config
│   ├── __init__.py
├── processing
│   ├── __init__.py
├── requirements.txt
├── runtime.txt
├── LICENSE
├── Procfile
├── README.md
└── wsgi.py
```
-->

## Directory Tree
```
|--   app
|     |-- Readme.md 
|     |-- notebooks -- main.py <-This is the main python jupyter notebook where execution of project starts
|     |-- data
|          |-- raw -- telecom_churn_data.csv <- This has input data for the project
|     |-- references <- This has Data dictionaries, manuals, and all other explanatory materials.
|     |-- reports <- Generated analysis as HTML, PDF, LaTeX, etc.
|           |-- figures <- Generated graphics and figures to be used in reporting
|     |-- presentations <-Manually created presentations for business users,stake holders in pptx,pdf etc
|           |-- images <- Manual images obtained from various sources for presentation & other checkpoints
```
