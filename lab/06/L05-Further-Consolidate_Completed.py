"""
# UW Data Science
# Now it's Your Turn to go further with the following exercise.
# Follow the instructions in the comments.
"""
# Lesson 6 Going Further - Consolidate Items into Groups
"""
Consider the following code in which you want to consolidate Device Names into 
# 4 group:
The first group is Phone, containing Cell Phone, Mobile Phone, and Phone
the second group is Appliance, containing Dish Washer, Refrigerator, and Oven
the third group is Computer, containing Laptop, Server, and Computer
the fourth group is Tool, containing Drill, Saw, Nail Gun, and Screw Driver
"""

import pandas as pd

# Create a List
DeviceName = [
"Cell Phone", "Dish Washer", "Laptop", "Phone", "Refrigerator", "Server",
"Oven", "Computer", "Drill", "Server", "Saw", "Computer", "Nail Gun",
"Screw Driver", "Drill", "Saw", "Saw", "Laptop", "Oven", "Dish Washer",
"Oven", "Server", "Mobile Phone", "Cell Phone", "Server", "Phone"]
# Create a dataframe
Device = pd.DataFrame(DeviceName, columns=["Name"])

#Display the resulting dataframe
Device #Answer

#####################
# List unique items
Device.loc[:,"Name"].unique()

# How many unique names are there?
Device.loc[:,"Name"].unique().shape #Answer

# Get the counts for each name value.
# Hint: see code in CategoryConsolidate.py
Device.loc[:,"Name"].value_counts() #Answer

# Plot the counts for each device name
# Hint: see code in CategoryConsolidate.py
Device.loc[:,"Name"].value_counts().plot(kind='bar') #Answer

#####################
# Use the following pattern to create a new column that consolidates device 
# names into groups:
# Devices.loc[Devices.loc[:, ExistingColumn] == SpecificName, NewColumn] = GroupName
# In this example, ExistingColumn is "Name", NewColumn is "Group", 
# GroupName is either "Appliance", "Computer", "Phone", or "Tool"
# and SpecificName is one of the specific device names found in column "Name"

# Create a new column
Device["Group"] = None

# Assign the GroupName, "Phone", to the NewColumn "Group" wherever there is
# the SpecificName "Cell Phone" in the ExistingColumn, "Name".
Device.loc[Device.loc[:, "Name"] == "Cell Phone", "Group"] = "Phone" #Answer

#Display the counts in the NewColumn
Device.loc[:,"Group"].value_counts() #Answer
#####################
# Consolidating one at a time will be tedious. Let's iterate through lists.

# Create lists of the consolidated device names
# Each list is a group
Appliance = ["Dish Washer", "Refrigerator", "Oven"]
Computer = ["Computer", "Laptop", "Server"]
Phone = ["Phone", "Cell Phone", "Mobile Phone"]
Tool = ["Drill", "Saw", "Nail Gun", "Screw Driver"]

# Check the first entry in the Appliance list
Appliance[0] #Answer

# Iterate for each item in the Appliance list and check the value counts
for x in Appliance:
    Device.loc[Device.loc[:,"Name"] == x , "Group"] = "Appliance"
    
#Display the counts in NewColumnn
Device.loc[:,"Group"].value_counts() #Answer

# Correct the following dictionary by replacing "???" with appropriate keys
# and values
DeviceGroup = { #Answer
        "Phone":Phone,
        "Appliance":Appliance,
        "Computer":Computer,
        "Tool":Tool
        }

#Iterate through the categories (keys)
for cat in DeviceGroup.keys():
    for DeviceName in DeviceGroup[cat]:
        Device.loc[Device.loc[:,"Name"] == DeviceName , "Group"] = cat

# Print out the dataframe
print(Device) #Answer

# Check the value counts
Device.loc[:,"Group"].value_counts() #Answer

# Print the plot of value counts for the Category
print(Device.loc[:,"Group"].value_counts().plot(kind='bar')) #Answer
