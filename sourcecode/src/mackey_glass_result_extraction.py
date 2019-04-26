import ast

"""
    Helper script for mackey_glass_prediction3.py

    Extracts results from JSON and converts them into 
    a format suitable for LaTex table.
"""
if __name__ == "__main__":
    lines = open("result_mgprediction.txt").readlines()
    for line in lines:
        curr_dict = ast.literal_eval(line)
        print("{} & {} & {} & {:0.4f} \\\\".format(curr_dict["name"],
            curr_dict["noise"], curr_dict["sigma"], curr_dict["rmse"]))
        