import numpy as np
import os
import re
import sys
import csv
import math
import copy
# import argparse
import subprocess
from functools import partial

def rewrite_predicate(predicate, parenthesis=True):
    if type(predicate[1]) is bool:
        if predicate[1]:
            predicate_str = str(predicate[0]) + "\n"
        else:
            predicate_str = "(not(" + str(predicate[0]) + ")\n"
    else:
        predicate_str = ""
        for proposition in predicate:
            predicate_str += str(proposition) +" "
    if parenthesis:
        predicate_str = "(" + predicate_str
        predicate_str += ")\n"
    predicate_str = "\t" + predicate_str
    return predicate_str

def generate_prob_hddl(hddl_dir, state, task, filename: str = "problem"):
    generate_problem_hddl(hddl_dir, state, task, filename=filename)

def generate_domain_hddl(hddl_dir, state, new_item, filename: str = "domain"):
    filename = hddl_dir + os.sep + filename + ".hddl"
    with open(filename, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    new_item_str_list = []
    for i in new_item:
        item = i.replace("''","")
        new_item_str_list.append(f"\t" + str(item) + f" - physobj\n")

    for item in new_item_str_list:
        if item in all_lines:
            new_item_str_list.remove(item)
     
    for line_no, line_text in enumerate(all_lines):
        if ":types" in line_text:
            for item in new_item_str_list:
                all_lines.insert(line_no+1, item)
            break

    with open(filename, "w", encoding="utf-8") as f:
        all_lines = "".join(all_lines)
        # print ("all_lines = ", all_lines)
        f.write(all_lines)

def generate_problem_hddl(hddl_dir, state, task, filename: str = "problem"):

    filename = hddl_dir + os.sep + filename + ".hddl"

    hder = _generate_header_prob()
    objs = _generate_objects(state)
    goals = _generate_goals(state, task)
    ints = _generate_init(state)

    hddl = "\n".join([hder, objs, goals, ints, ")\n"])
    with open(filename, "w", encoding="utf-8") as f:
        f.write(str(hddl))
        # print(f"Problem hddl written to {filename}.")


def _generate_header_prob():
    return f"(define\n\t(problem adeGeneratedProblem)" + "\n\t" + f"(:domain adeGeneratedDomain)"

def _generate_objects(state):
    objects_list = []
    for item in state.items:
        objects_list.append("\t\t"+item[0]+" - "+item[1])
    objs = "\n".join(objects_list)
    return "\n".join(["(:objects", objs, "\t)"])

def _generate_init(state):
    init_list = map(rewrite_predicate, state.grounded_predicates)
    ints = "".join(init_list)
    return "\n".join(["(:init", ints, "\t)"])

def _generate_goals(state, task):
    return "".join(["(:htn\n 	:parameters ()\n\t:ordered-subtasks (and (",task,")) \n)"])
 
    