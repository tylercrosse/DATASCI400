"""
Lesson 01 Assignment

Instructions:
Craft a simple Python script that contains a function that will output your name and date.
The script will also execute the function.

Create a new Python script that includes the following:

Function that returns your name, and call to that function
Code that prints the current date and time with:
Import statement
Print statement(s)
Comment with source citation for date code including URL if applicable.
"""

from datetime import datetime

def print_name():
    """ prints my name """
    print('Tyler Crosse')

print_name()

def print_current_time():
    """
    prints the current time, without formatting
    https://docs.python.org/3.6/library/datetime.html
    """
    current_time = datetime.now()
    print(current_time)

print_current_time()
