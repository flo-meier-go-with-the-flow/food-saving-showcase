# Ideas

Train a model for every Weekday

# Todos

make model from num people at 11:30 to menus_sold
make pickle, load pickle
https://pemagrg.medium.com/saving-sklearn-model-to-pickle-595da291ec1c
use this to create a prediction together with sarimax model for num menus sold.


Make a dataframe that contains the labels and relevant features:
 - day in month
 - separate Spring and Autumn
 - num people leaving for lunch elsewhere on previous days/weeks
 - Num people enter before 10 binned by hour

# App
- Flask
- Download data everyday at 11:30
  - Check out sched
  - https://docs.python.org/3/library/sched.html
  - Check out cronitor
  - https://crontab.guru/#0_12_*_*_*
  - Check out GCP Cloud scheduler
  - https://cloud.google.com/scheduler/docs/schedule-run-cron-job


# Bug Fixing

2022-09-27 is droped because of NAN values in weather data

# Learnings
## What can we predict given num peopele at 10?
num people at 11:30: 
rmse 5.3
mae 4.2
main feature ist num people at 10 additional features like Weekday affect the scores by +-0.1

num people at 12:20: 
rmse 8.8
mae 6.9
main feature ist num people at 10 additional features like Weekday affect the scores by +-0.1


num people at 12:20: 
rmse 22.7
mae 16.3



# Todo
- plots for Markus
- Bestellmengen vorausagen:
  - 1 bis 7 Tage


# Was brauchen wir von Markus:
- Salatdaten, Kassensystemdaten
- Menustatistik:
  - Busenesslunch
  - Genauigkeit?, Gibt es Fehlerquellen?
- Menuliste
- Foodwaste messen?

