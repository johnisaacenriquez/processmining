def make_initial_caseid(event, process_model):
    import random
    caseid = []
    cases = []
    counter = 1
    enabled_a = []
    enabled = []
    double_B = 0
    for i in range(len(event)):
        if event[i] == "A":
            caseid.append(counter)
            counter += 1
            cases.append(list(event[i]))
        else:
            case_index = []
            for j in range(len(cases)):
                for key in process_model:
                    if event[i] == "B":
                        if cases[j][-1] == "B":
                            cases[j].pop()
                            double_B = 1
                    if cases[j] + [event[i]] == process_model[key][:len(cases[j])+1]:
                        enabled_a.append(key)
                        case_index.append(j)
                        if double_B == 1:
                            cases[j].append("B")
                        double_B = 0
                enabled.append(enabled_a)
                enabled_a = []
            if len([i for i in enabled if i]) > 1:
                index = random.randint(min(case_index),max(case_index))
            else:
                index = next((u for u,v in enumerate(enabled) if v), 1000000)
                if index == 1000000:
                    index = random.randint(min(caseid)-1,max(caseid)-1)
            enabled = []
            caseid.append(index+1)
            cases[caseid[-1]-1].append(event[i])
    return(caseid)

def make_candidate_caseid(event, process_model, caseid_current, iteration):
    import random
    s = iteration

    # Set up input  parameters

    # Copy event up to the element defined by iteration number
    event_list = event[:s]

    # Count the number of A events
    counter = event_list.count("A")

    # Copy case id up to the element defined by iteration number
    caseid_old = caseid_current[:s]

    # Make a list of events separated by case id
    cases_old = []
    for j in range(len(caseid_old)):
        count = 0
        if event[j] == "A":
            cases_old.append(list(event[j]))
            count += 1
        else:
            if len(cases_old) > 1:
                cases_old[caseid_old[j]-1].append(event[j])
            else:
                cases_old.append(event[j])


    # Main loop  for making candidate case id:
    enabled  = []
    enabled_a = []
    double_B = 0
    caseid = caseid_old
    cases = cases_old

    for i in range(s, len(event)):

        # Take care of cyclic event B in cases
        for j in range(len(cases)):
            if cases[j][-1]=="B":
                if cases[j][-2] == "B":
                    cases[j].pop()

        # Start a new case if event is A
        if event[i] == "A":
            caseid.append(counter+1)
            counter += 1
            cases.append(list(event[i]))
        else:
            case_index = []
            for j in range(len(cases)):
                for key in process_model:
                    # Algorithm to deal with cyclic event B
                    if event[i] == "B":
                        if cases[j][-1] == "B":
                            cases[j].pop()
                            double_B = 1
                    if cases[j] + [event[i]] == process_model[key][:len(cases[j])+1]:
                        enabled_a.append(key)
                        case_index.append(j)
                        if double_B == 1:
                            cases[j].append("B")
                        double_B = 0
                enabled.append(enabled_a)
                enabled_a = []
            if len([i for i in enabled if i]) > 1:
                index = random.randint(min(case_index),max(case_index))
            else:
                # 1000000 is placed to handle exeption
                index = next((u for u,v in enumerate(enabled) if v), 1000000)
                if index == 1000000:
                    index = random.randint(min(caseid)-1,max(caseid)-1)
            enabled = []
            caseid.append(index+1)
            cases[caseid[-1]-1].append(event[i])
    return caseid
