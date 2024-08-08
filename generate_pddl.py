'''
# authors: Pierrick Lorang
# email: pierrick.lorang@tufts.edu
'''
import os
import numpy as np
import re

pddl_dir = "./PDDL"

def restrict_pddl(action_name, object_name, domain_name="domain", problem_name="problem", modified_domain_name="modified_domain", modified_problem_name="modified_problem"):
    # Open the problem file and read its contents
    problem_path = pddl_dir + os.sep + problem_name + ".pddl"
    with open(problem_path, "r") as file:
        problem_content = file.read()
    # Open the domain file and read its contents
    domain_path = pddl_dir + os.sep + domain_name + ".pddl"
    with open(domain_path, "r") as file:
        domain_content = file.read()

    # Get the type of the object
    object_type = get_object_type(object_name, problem_content)
    if not object_type:
        raise ValueError(f"Object '{object_name}' not found in problem file")

    first_var = get_first_variable_of_action(action_name, domain_content)

    # Create a new type that excludes the specified object
    new_type = f"{object_type}_except_{object_name}"

    # Replace the object type in the action with the new type
    action_pattern = f"\(:action {action_name}.*?:parameters \(\?{first_var} - \w+"
    modified_domain_content = re.sub(action_pattern, f"(:action {action_name}\n  :parameters (?{first_var} - {new_type}", domain_content, flags=re.DOTALL)
    # Add the new type to the types section of the domain as a subset of the original type
    types_pattern = f"\(:types.*?{object_type}"
    modified_domain_content = re.sub(types_pattern, f"(:types\n  {new_type} - {object_type} {object_type}", modified_domain_content, flags=re.DOTALL)

    # Write the modified domain content to a new file
    modified_domain_path = pddl_dir + os.sep + modified_domain_name + ".pddl"
    with open(modified_domain_path, "w") as file:
        file.write(modified_domain_content)

    # Replace the type of all objects except the specified one with the new type
    objects_pattern = f"(\s+)(?!{object_name})(\w+ - {object_type})"
    modified_problem_content = re.sub(objects_pattern, f"\\1\\2 - {new_type}", problem_content)

    # Correct the replacement to ensure only the type is replaced, not the object name
    modified_problem_content = re.sub(f" - {object_type} - {new_type}", f" - {new_type}", modified_problem_content)

    # Write the modified problem content to a new file
    modified_problem_path = pddl_dir + os.sep + modified_problem_name + ".pddl"
    with open(modified_problem_path, "w") as file:
        file.write(modified_problem_content)

def get_first_variable_of_action(action_name, domain_content):
    # Pattern to match the action and its first variable
    pattern = f"\(:action {action_name}.*?:parameters \(\?(\w+)"
    match = re.search(pattern, domain_content, flags=re.DOTALL)
    return match.group(1) if match else None

def get_object_type(object_name, problem_content):
    # Pattern to match the object and its type
    pattern = f"{object_name} - (\w+)"
    match = re.search(pattern, problem_content)
    return match.group(1) if match else None

def remove_action(action_name, domain_name="domain", modified_domain_name="modified_domain"):
    # Open the domain file and read its contents
    domain_path = pddl_dir + os.sep + domain_name + ".pddl"
    with open(domain_path, "r") as file:
        content = file.read()

    # Define the pattern to find the action
    pattern = f"\(:action {action_name}.*?:effect.*?\)\n"

    # Use regex to remove the action
    modified_content = re.sub(pattern, "", content, flags=re.DOTALL)

    modified_path = pddl_dir + os.sep + modified_domain_name + ".pddl"
    # Write the modified content to a new file
    with open(modified_path, "w") as file:
        file.write(modified_content)

def rewrite_predicate(predicate, parenthesis=True, jump_line=True, write_negations=False):
    key, value = predicate
    key = key.replace(",", " ").replace("(", " ").replace(")", " ")  # Replace commas and parentheses with spaces
    # If type value is string and is False or True, then convert it to boolean
    if type(value) is str:
        if value == "False":
            value = False
        elif value == "True":
            value = True
    if type(value) == bool or type(value)==np.bool_:
        if value:
            predicate_str = str(key) + "\n"
        else:
            if write_negations:
                predicate_str = "not (" + str(key) + ")\n"
            else:
                return ""
    else:
        predicate_str = ""
        for proposition in key:
            predicate_str += str(proposition)
    if parenthesis:
        predicate_str = predicate_str.strip()  # Remove leading/trailing spaces before adding parentheses
        predicate_str = "(" + predicate_str + ")"
        predicate_str += "\n"
    predicate_str = "\t" + predicate_str
    if not jump_line:
        predicate_str = predicate_str.replace("\n", "")
        predicate_str = predicate_str.replace("\t", "")
    return predicate_str

def generate_prob_pddl(pddl_dir, init, goal, filename: str = "problem"):
    generate_problem_pddl(pddl_dir, init, goal, filename=filename)

def generate_domain_pddl(pddl_dir, new_item, filename: str = "domain"):
    filename = pddl_dir + os.sep + filename + ".pddl"
    with open(filename, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    new_item_str_list = []
    for i in new_item:
        item = i.replace("''","")
        new_item_str_list.append(f"\t" + str(item) + f" - object\n")

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

def generate_problem_pddl(pddl_dir, state, goal, filename: str = "problem"):

    filename = pddl_dir + os.sep + filename + ".pddl"

    hder = _generate_header_prob()
    objs = _generate_objects(state)
    ints = _generate_init(state)
    goals = _generate_goals(state, goal)

    pddl = "\n".join([hder, objs, ints, goals, "\n"])
    with open(filename, "w", encoding="utf-8") as f:
        f.write(str(pddl))
        # print(f"Problem pddl written to {filename}.")

def _generate_header_prob():
    return f"(define (problem generatedProblem)" + "\n\t" + f"(:domain generatedDomain)"

def _generate_objects(state):
    objects_list = []
    for item in state.entities:
        objects_list.append("\t\t"+item[0]+" - "+item[1])
    objs = "\n".join(objects_list)
    return "\n".join(["(:objects", objs, "\t)"])

def _generate_init(state):    
    init_list = map(rewrite_predicate, state.grounded_predicates.items())
    ints = "".join(init_list)
    return "\n".join(["(:init", ints, "\t)"])

def _generate_goals(state, goal):
    return "".join(["(:goal (and ",goal,")) \n)"])