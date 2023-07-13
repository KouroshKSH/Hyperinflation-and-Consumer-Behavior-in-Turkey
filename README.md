<p align="center">
  <img src="https://github.com/KouroshKSH/Hyperinflation-and-Consumer-Behavior-in-Turkey/blob/master/images/banner.png">
</p>
 
# Hyperinflation and Consumer Behavior in Turkey
The aim of this project is to investigate the effects of hyperinflation on consumer behavior in the food and beverage industry in Turkey using time series and sentiment analysis.

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter">
</p>

---

## Team Members
The team members of this project are:
- Arya Hassibi ([LinkedIn](https://www.linkedin.com/in/aryahassibi/), [GitHub](https://github.com/aryahassibi))
- Milad Bafarassat ([LinkedIn](https://www.linkedin.com/in/miladbafarassat/), [GitHub](https://github.com/Miladbaf))
- Kourosh Sharifi ([LinkedIn](https://www.linkedin.com/in/kouroshsharifi/), [GitHub](https://github.com/KouroshKSH/))

> **Note:** As the time of making this project, all the members are Sophomore Computer Science students at Sabanci University, under the supervision of [Dr. Onur Varol](http://www.onurvarol.com/).

---

## Research Question
How has hyperinflation in Turkey affected consumer behavior in the food and beverage industry, specifically in terms of restaurant and cafe preferences and spending habits?

## Hypothesis
The current hyperinflation period in Turkey has led to significant changes in consumer behavior in the food and beverage industry, with people spending more money on dining out and showing a preference for certain types of restaurants and cafes.

---

## Data Sources
To investigate the research question, the following data sources could be used:
1. **TÜİK:** TÜİK is the Turkish Statistical Institute that provides a vast amount of data on food and beverage industry revenues, consumer spending habits, retail sales, and inflation rates in Turkey. A number of datasets are acquired from TUIK, especifically the various price index datasets that hold the numbers for the food and beverage industry, consumer expenditure, household expenditure in each region, and others.
2. **Yemeksepeti:** Yemeksepeti is a Turkish food ordering platform, also known as the equivalent of Uber Eats or DoorDash. Analyzing data from Yemeksepeti can provide insights into consumer preferences and spending habits for different types of food and beverage establishments. These datasets can be accessed through [Google Drive](https://drive.google.com/drive/folders/1l4J1IXDtvGCOBzbD7jX-Y-Kud4FOj86S?usp=sharing).

## Methods
To investigate the research question, the following methods could be used:
1. **Time series analysis:** This method could be used to track changes in consumer behavior over time and identify trends in restaurant and cafe preferences and spending habits. It can also be used to forecast future behavior of the consumers. In the context of our research question, time series analysis can be used to track changes in consumer behavior in the food and beverage industry over time, including changes in spending habits, restaurant and cafe preferences, and the impact of hyperinflation on these trends. For this purpose, ARIMA and SARIMA forecasting models were implemented.

2. **Sentiment analysis:** This method could be used to analyze online reviews and ratings for different types of food and beverage establishments to gain insights into consumer attitudes and perceptions. A number of libraries are suitable for this part, such as Scikit-learn, NLTK, spaCy and others, which can help with tasks such as tokenizing text, identifying parts of speech, analyzing sentiment polarity, and more.

3. **Data visualization:** This method can be used to present the findings of the analysis in a visual form and to communicate the insights to the stakeholders. Libraries such as Matplotlib and Seaborn can be utilized for this task. Data visulization also helps with the exploratory data analysis phase of the project as well since it provides insights through graphical representation of the data gathered, which could have been overlooked via tabular representation.

---

## Graphics
The plots below are just some of the findings of this project. For more information, please visit the Jupyter notebooks of the project to get a better understanding of the different phases of the project, and how each part was carried out.

### Consumer Price Index
![](https://github.com/KouroshKSH/Hyperinflation-and-Consumer-Behavior-in-Turkey/blob/master/images/cpi_eda_months_2005-2023.png)

### Distribution of CPI
![](https://github.com/KouroshKSH/Hyperinflation-and-Consumer-Behavior-in-Turkey/blob/master/images/cpi_boxplot_eda_2005-2023.png)

### Time Series Decomposition of CPI
![](https://github.com/KouroshKSH/Hyperinflation-and-Consumer-Behavior-in-Turkey/blob/master/images/cpi_decomp_2005-2019.png)

### Autocorrelation of Food and Services Price Index
![](https://github.com/KouroshKSH/Hyperinflation-and-Consumer-Behavior-in-Turkey/blob/master/images/fspi_autocorr_plots.png)

### SARIMA Forecasts
![](https://github.com/KouroshKSH/Hyperinflation-and-Consumer-Behavior-in-Turkey/blob/master/images/fspi_accom_sarima_2017-2023.png)

---

## Conclusion
The results of this research support the hypothesis that Turkish people are in fact spending more on non-durable goods during the hyperinflation that the country is experiencing. 

---

## License
This repository is licensed under the [MIT License](https://opensource.org/license/mit/).

<p align="center">
  <img src="https://img.shields.io/pypi/l/ansicolortags.svg" />
</p>
