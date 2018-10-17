# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 16:53:16 2018

@author: v-jopre
"""

## Creating a function which prints my name

def NamePrint(name):
    return(name)

NamePrint(name = "Joseph Prestifilippo")   
print(NamePrint(name = "Joseph Prestifilippo"))

##Code that prints current date and time
import datetime as d

current_iso = d.datetime.now().isoformat()

other_date_format = d.datetime.now().strftime("%Y-%m-%d %H:%M")

print(current_iso)

print(other_date_format)

##URL Source for import and date info
##https://www.saltycrane.com/blog/2008/06/how-to-get-current-date-and-time-in/
