from flask import Flask
from flask import request, make_response, render_template
from datetime import date, datetime, time
import tools




app = Flask('super-app')
@app.route('/')
def index():
   dic={
   'day_date_time':tools.display_day_time(),
   'num_people_in_building':"in construction",
   'next_relevant_event':tools.next_relevant_event(),
   'prediction_number_people':'in construction',
   'prediction_number_menus_sold':'in construction'
   }
   return render_template('forecast.html',**dic)




app.run()