def make_process_model():
    model_1  = ["A", "C", "X"]
    model_2 = ["A", "B", "C", "X"] # Event "B" can be repeated
    model_3 = ["A", "B", "D", "X"] # Event "B" can be repeated
    #process_model = [model_1, model_2, model_3]
    process_model = {"model_1":model_1, "model_2":model_2, "model_3":model_3}
    return process_model

def make_process_model_list():
    model_1  = ["A", "C", "X"]
    model_2 = ["A", "B", "C", "X"] # Event "B" can be repeated
    model_3 = ["A", "B", "D", "X"] # Event "B" can be repeated
    process_model_list = [model_1, model_2, model_3]
    return process_model_list