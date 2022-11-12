import numpy as np
from datetime import datetime
from datetime import date
from datetime import time
from datetime import timedelta
import pandas as pd
import copy

def make_logarray(caseid):
    # Define a list of unique events excluding the first event ("A")
    event_unique = ["B", "C", "D"]

    # Make a list of event id and timestamp
    event = ["A", "A", "B", "B", "C", "D", "A", "C"]
    timestamp = ['13:01:01', '13:02:02', '13:05:03', '13:07:04', '13:15:05', '13:30:06', '13:52:07', '14:08:08']

    # Import to Pandas
    logarray_dict = dict(Event=event, Timestamp=timestamp)
    logarray = pd.DataFrame(logarray_dict)
    logarray["Case_ID"] = caseid
    # Separate events belonging to same case id
    case_1 = logarray.loc[logarray["Case_ID"] == 1].reset_index(drop=True)
    case_2 = logarray.loc[logarray["Case_ID"] == 2].reset_index(drop=True)
    case_3 = logarray.loc[logarray["Case_ID"] == 3].reset_index(drop=True)

    # Dictionary of separate events. Maybe useful later
    case_1_dict = case_1.to_dict()
    case_2_dict = case_2.to_dict()
    case_3_dict = case_3.to_dict()

    # List of dictionaries
    logarray = [case_1_dict, case_2_dict, case_3_dict]

    return logarray

def calculate_time_fitness(logarray):
    dt = []
    dt_list = {}

    eventlist = []
    for i in range(len(logarray)):
        for j in range(len(logarray[i]["Event"])):
            eventlist.append(list(logarray[i]["Event"].values())[j])
    event_unique = list(np.unique(eventlist))
    event_unique.remove("A")
    for i in range(0, len(event_unique)):
        # Initialize dt_list
        dt_list[i] = []
        for j in range(0, len(logarray)):
            case = pd.DataFrame(logarray[j])
            ac_case = case.loc[case["Event"] == event_unique[i]].index.to_list()
            for j in ac_case: 
                dt_list[i].append((pd.Timedelta(case["Timestamp"][j]) - pd.Timedelta(case["Timestamp"][j-1]))/timedelta(minutes=1))

    for i in range(0, len(dt_list.keys())):
        dt = dt + (((dt_list[i] - np.mean(dt_list[i]))**2).tolist())
    

    ft = (sum(dt) / len(eventlist))
    return round(ft, 3)

def calculate_alignment_fitness(logarray, process_model):
    case_events = []
    for i in range(0, len(logarray)):
        case_events.append(list(logarray[i]["Event"].values()))

    fa = []
    for i in range(0, len(case_events)):
        models = copy.deepcopy(process_model)

        # Match the length of case and model
        for j in range(len(models)):
            if len(case_events[i]) > len(models[j]):
                lendiff = len(case_events[i]) - len(models[j])
                for k in range(0, lendiff):
                    models[j].insert(-1, "X")

        # Alignment cost based on mismatch 
        fa_case = []           
        for j in range(0, len(models)):
            
            fa_case_model = 0.0
            for k in range(0, len(case_events[i])):
                if case_events[i][k] not in models[j][k]:
                    if case_events[i][k] == "B" and case_events[i][k-1] == "B":
                        models[j].insert(k, "B")
                    else:
                        fa_case_model += 1
                        models[j].insert(k, case_events[i][k])
            
            if models[j][len(case_events[i])] != "X":   
                fa_case_model += 1
            
            fa_case.append(fa_case_model)
        
        fa.append(fa_case)
    alignment_cost_list = []
    for i in range(len(fa)):
        alignment_cost_list.append(np.min(fa[i]))
    alignment_cost = np.sum(alignment_cost_list)
    return alignment_cost


    