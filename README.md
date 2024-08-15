# End-to-End-Analysis-and-Prediction-of-Loan-Approval-with-Deployment

This dataset ([source](https://www.kaggle.com/itssuru/loan-data)) consists of data from almost 10,000 borrowers that took loans - with some paid back and others still in progress. It was extracted from lendingclub.com which is an organization that connects borrowers with investors.

First some important imports and house-keeping:

```python
# Load packages
import numpy as np 
import pandas as pd
```

## PROBLEM STATEMENT 
---

The purpose is to gain useful insight from the available data and train a model to predict the probability of a loan not being paid in full

In the first part, the exploratory analysis will provide information and insights on the data. Tables and visualizations will be used to better understand the available data and answer relevant questions

In the second part, the data will be used to predict the probability of a loan not being paid in full. Different models will be train and the best-performing one will be selected.

### RESEARCH
---
*some information for context from:* https://www.capitalone.com/learn-grow/money-management/revolving-credit-balance/
#### How Does Revolving Credit Work?
If you’re approved for a revolving credit account, like a credit card, the lender will set a credit limit. The credit limit is the maximum amount you can charge to that account. When you make a purchase, you’ll have less available credit. And every time you make a payment, your available credit goes back up.

Revolving credit accounts are open ended, meaning they don’t have an end date. As long as the account remains open and in good standing, you can continue to use it. Keep in mind that your minimum payment might vary from month to month because it’s often calculated based on how much you owe at that time. 

#### What Is a Revolving Balance?
If you don’t pay the balance on your revolving credit account in full every month, the unpaid portion carries over to the next month. That’s called a revolving balance. 

You might apply for credit assuming you’ll always pay your balance in full every month. But real life can get in the way. Cars break down. Doctors’ appointments come up. And if you can’t pay your full balance, you’ll find yourself carrying a revolving balance to the following month. 

#### What About Revolving Balances and Interest?
As the Consumer Financial Protection Bureau (CFPB) explains, “A credit card’s interest rate is the price you pay for borrowing money.” And the higher your revolving balance, the more interest you might be charged. But you can typically avoid interest charges by paying your balance in full every month. 

#### What’s Revolving Utilization and How Does It Impact Credit Score?
Your credit utilization ratio—sometimes called revolving utilization—is how much available credit you have compared with the amount of credit you’re using. According to the CFPB, you can calculate your credit utilization ratio by dividing your total balances across all of your accounts by your total credit limit.

So why does your credit utilization ratio matter? It’s one of the factors that determines your credit score. If you manage credit responsibly and keep your utilization ratio relatively low, it might help you improve your credit score. The CFPB recommends keeping your utilization below 30% of your available credit. 
