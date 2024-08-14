#import pddlpy
import re
from collections import namedtuple

class Action:
    def __init__(self, parameters, preconditions, effects, numerical_preconditions=None, numerical_effects=None, function_effects=None, cost=None, name=None):
        self.parameters = parameters
        self.preconditions = preconditions
        self.effects = effects
        self.numerical_preconditions = numerical_preconditions if numerical_preconditions is not None else {}
        self.numerical_effects = numerical_effects if numerical_effects is not None else {}
        self.function_effects = function_effects if function_effects is not None else {}
        self.cost = cost
        self.name = name if name != None else "a0"
        self._to_pddl()

    # def __eq__(self, other):
    #     if isinstance(other, Action):
    #         if (other.numerical_effects == {} and other.function_effects == {} and other.effects != {}) or (self.numerical_effects == {} and self.function_effects == {} and self.effects != {}):
    #             # Compare preconditions, effects, numerical_effects and numerical_preconditions
    #             equal = (self.preconditions == other.preconditions and
    #                     self.effects == other.effects and
    #                     self.numerical_effects == other.numerical_effects and
    #                     self.numerical_preconditions == other.numerical_preconditions)
    #         else:
    #             # Compare preconditions, effects, and numerical_effects only
    #             equal = (self.preconditions == other.preconditions and
    #                     self.effects == other.effects and
    #                     self.numerical_effects == other.numerical_effects)

    #         # Compare function_effects, ignoring 'total-cost'
    #         self_function_effects = {k: v for k, v in self.function_effects.items() if k != 'total-cost'}
    #         other_function_effects = {k: v for k, v in other.function_effects.items() if k != 'total-cost'}
    #         equal = equal and self_function_effects == other_function_effects

    #         return equal

    #     return False

    def __eq__(self, other):
        if isinstance(other, Action):
            # get a list of the types of a0.used_parameters
            self_types = [type for param, type in self.used_parameters]
            other_types = [type for param, type in other.used_parameters]
            # convert it to a string separated by spaces
            self_grounding = self.name + ' ' + ' '.join(self_types)
            other_grounding = other.name + ' ' + ' '.join(other_types)
            # ground the actions
            self_ground_with_types = self.ground_action(action_string=self_grounding)
            other_ground_with_types = other.ground_action(action_string=other_grounding)
            # Compare preconditions and effects
            equal = self_ground_with_types.preconditions == other_ground_with_types.preconditions
            equal = equal and self.effects == other.effects
            
            # Compare numerical_effects, ignoring 'total-cost'
            #self_numerical_effects = {k: v for k, v in self.numerical_effects.items() if k != 'total-cost'}
            #other_numerical_effects = {k: v for k, v in other.numerical_effects.items() if k != 'total-cost'}
            #equal = equal and self_numerical_effects == other_numerical_effects

            # Compare function_effects, ignoring 'total-cost'
            self_ground_with_types_function_effects = {k: v for k, v in self_ground_with_types.function_effects.items() if k != 'total-cost'}
            other_ground_with_types_function_effects = {k: v for k, v in other_ground_with_types.function_effects.items() if k != 'total-cost'}
            equal = equal and self_ground_with_types_function_effects == other_ground_with_types_function_effects

            return equal

        return False

    def _cheaper_cost_(self, other):
        if other == self:
            if self.function_effects['total-cost'] < other.function_effects['total-cost']:
                self.function_effects = other.function_effects
                return True
        return False
    
    def _same_effects_(self, other):
        same_effects = True
        # Check whether the effects, numerical_effects and function_effects are the same
        if self.effects != other.effects or self.numerical_effects != other.numerical_effects or self.function_effects != other.function_effects:
            same_effects = False
        return same_effects
    
    def _is_weaker_(self, other):
        # Check whether the preconditions are less constraining (weaker) than the other action
        weaker = True
        for predicate, value in self.preconditions.items():
            if predicate not in other.preconditions.keys():
                weaker = False
                break
            elif value != other.preconditions[predicate]:
                weaker = False
                break
        return weaker

    def _to_pddl(self):
        # Transform to PDDL format
        preconditions = []
        for predicate, value in self.preconditions.items():
            if "(" in predicate and ")" in predicate:
                formatted_predicate = f'({predicate.split("(")[0]} {" ".join(predicate.split("(")[1][:-1].split(",")).replace("  ", " ")})'
            else:
                formatted_predicate = f'({predicate})'
            if value:
                preconditions.append(formatted_predicate)
            else:
                preconditions.append(f'(not {formatted_predicate})')
        preconditions = ' '.join(preconditions)

        effects = []
        for predicate, value in self.effects.items():
            if "(" in predicate and ")" in predicate:
                formatted_predicate = f'({predicate.split("(")[0]} {" ".join(predicate.split("(")[1][:-1].split(",")).replace("  ", " ")})'
            else:
                formatted_predicate = f'({predicate})'
            if value:
                effects.append(formatted_predicate)
            else:
                effects.append(f'(not {formatted_predicate})')
        effects = ' '.join(effects)

        if self.numerical_preconditions != {}:
            numerical_preconditions = ' '.join([f'(= ({predicate.replace(",", " ").replace("(", " ").replace(")", " ")}) {value})' for predicate, value in self.numerical_preconditions.items()])
        else:
            numerical_preconditions = ''
        if self.numerical_effects != {}:
            numerical_effects = ' '.join([f'(= ({predicate.replace(",", " ").replace("(", " ").replace(")", " ")}) {value})' for predicate, value in self.numerical_effects.items()])
        else:
            numerical_effects = ''
        if self.function_effects != {}:
            function_effects = ' '.join([f'(increase ({predicate.replace(",", " ").replace("(", " ").replace(")", " ")}) {value})' if int(value) > 0 else f'(decrease ({predicate.replace(",", " ").replace("(", " ").replace(")", " ")}) {-value})' for predicate, value in self.function_effects.items()])
        else:
            function_effects = ''

        # FOR NOW it does not account for numerical preconditions (so that the agent does not need to know location requirements for actions)
        if self.effects == {}:# and self.numerical_effects == {}:
            numerical_preconditions = ''
        
        used_parameters = []
        for param, type in self.parameters:
            param_str = f'?{param}'
            if param_str in preconditions or param_str in effects or param_str in numerical_preconditions or param_str in numerical_effects or param_str in function_effects:
                used_parameters.append((param, type))
        self.used_parameters = used_parameters
        parameters = ' '.join([f'?{param} - {type}' for param, type in used_parameters])


        return f'(:action {self.name}\n  :parameters ({parameters})\n  :precondition (and {preconditions} {numerical_preconditions})\n  :effect (and {effects} {function_effects} {numerical_effects}))\n'

    def ground_action(self, action_string):
        action_parts = action_string.split(' ')
        name = action_parts[0]
        parameters = [(part, type) for part, type in zip(action_parts[1:], [param[1] for param in self.used_parameters])]

        # Create a mapping from parameters to placeholders
        param_to_placeholder = {f'?{placeholder[0]}': param[0] for param, placeholder in zip(parameters, self.used_parameters)}
        # print("\nDEBUGGING")
        # print(self._to_pddl())
        # print(action_parts)
        # print(parameters)
        # print(self.used_parameters)
        # print(self.parameters)
        # print(param_to_placeholder)
        # Ground the preconditions and effects
        grounded_preconditions = {self._replace_placeholders(predicate, param_to_placeholder): value
                                for predicate, value in self.preconditions.items()}
        grounded_numerical_preconditions = {self._replace_placeholders(predicate, param_to_placeholder): value for predicate, value in self.numerical_preconditions.items()}
        grounded_effects = {self._replace_placeholders(predicate, param_to_placeholder): value
                            for predicate, value in self.effects.items()}
        grounded_numerical_effects = {self._replace_placeholders(predicate, param_to_placeholder): value for predicate, value in self.numerical_effects.items()}
        grounded_function_effects = {self._replace_placeholders(predicate, param_to_placeholder): value for predicate, value in self.function_effects.items()}

        return Action(parameters, grounded_preconditions, grounded_effects, numerical_preconditions=grounded_numerical_preconditions ,numerical_effects=grounded_numerical_effects, function_effects=grounded_function_effects, name=name)

    def _replace_placeholders(self, predicate, param_to_placeholder):
        for placeholder, param in param_to_placeholder.items():
            predicate = predicate.replace(placeholder, param)
        return predicate.replace(' ', '')

    def __str__(self):
        return self._to_pddl()

def load_action_from_file(domain_file, problem_file, numerical):

    actions = []
    try:
        domprob = pddlpy.DomainProblem(domain_file, problem_file)
        for operator in domprob.operators():
            operator_details = domprob.domain.operators[operator]
            parameters = list(operator_details.variable_list.items())
            parameters = [(param[0].replace('?', ''), param[1]) for param in parameters]
            preconditions = operator_details.precondition_pos
            neg_preconditions = operator_details.precondition_neg
            effects = operator_details.effect_pos
            neg_effects = operator_details.effect_neg
            effects = [atom.predicate for atom in list(effects)]
            neg_effects = [atom.predicate for atom in list(neg_effects)]
            preconditions = [atom.predicate for atom in list(preconditions )]
            neg_preconditions = [atom.predicate for atom in list(neg_preconditions )]
            effects = {f'{effect[0]}({", ".join(effect[1:])})': True for effect in list(effects)}
            effects.update({f'{effect[0]}({", ".join(effect[1:])})': False for effect in neg_effects})
            pre = {f'{pre[0]}({", ".join(pre[1:])})': True for pre in preconditions}
            pre.update({f'{pre[0]}({", ".join(pre[1:])})': False for pre in neg_preconditions})
            actions.append(Action(parameters, pre, effects, name=operator))
    except Exception as e:
        print("Error loading actions from domain and problem files: ", e)
    return actions
    
def restrict_action(actions, grounded_action):
    # Parse the grounded action
    grounded_action_name = grounded_action.split('(')[0]
    grounded_action_params = grounded_action.split('(')[1][:-1].split(', ')

    # Create a new list of actions
    new_actions = []

    # Iterate over the actions
    for action in actions:
        # If the action name matches the grounded action name
        if action.name == grounded_action_name:

            # Iterate over the parameters of the action
            for i, (param, type) in enumerate(action.parameters):
                # Create a copy of the action
                new_action = Action(action.parameters.copy(), action.preconditions.copy(), action.effects.copy(), name=action.name + str(i))

                # Add a new precondition that the parameter cannot have that value
                new_action.preconditions[f'= ?{param} {grounded_action_params[i]}'] = False

                # Add the new action to the list of new actions
                new_actions.append(new_action)
        else:
            # If the action name does not match the grounded action name, add the action to the list of new actions without modifying it
            new_actions.append(action)

    # Return the new list of actions
    return new_actions

def replace_actions_in_domain(file, new_file_path, actions):
    with open(file, 'r') as f:
        domain = f.read()
    #print(domain)
    start, rest = domain.split('(:predicates', 1)
    start, constants = start.rsplit('(:constants', 1)
    predicates = '(:predicates' + rest
    constants = '(:constants' + constants

    domain = start + constants  + predicates

    try:
        start, end = domain.split('(:action', 1)
        end = ')'+end.rsplit(')', 1)[1]
    except:
        start, end  = domain.split('()', 1)
        end = ')'+end.rsplit(')', 1)[1]

    new_actions = '\n'.join([action._to_pddl() for action in actions])
    #print("New Actions = ", new_actions)

    new_file = new_file_path
    with open(new_file, 'w') as f:
        f.write(start + new_actions + end)

def extract_objects_from_predicate(predicate):
    if '(' in predicate and ')' in predicate:
        objects = predicate.split('(')[1].split(')')[0].split(',')
        return [obj.strip() for obj in objects if obj.strip()]  # remove empty strings and strip whitespace
    else:
        return []  # return an empty list if the predicate does not take any objects
    
def replace_whole_word(text, old_word, new_word):
    return re.sub(r'\b' + old_word + r'\b', new_word, text)

def operator_learner_minimal_effects_ungrounded(predicates_t, predicates_t1, type_mapping, name=None):
    parameters = []
    preconditions = {}
    effects = {}

    for predicate, value in predicates_t.items():
        objects = extract_objects_from_predicate(predicate)
        parameters.extend([(obj, type_mapping[obj]) for obj in objects])
        parametrized_predicate = predicate
        for obj in objects:
            parametrized_predicate = replace_whole_word(parametrized_predicate, obj, '?' + obj)
        preconditions[parametrized_predicate] = value

    for predicate, value in predicates_t1.items():
        if predicate not in predicates_t or predicates_t1[predicate] != predicates_t[predicate]:
            objects = extract_objects_from_predicate(predicate)
            parameters.extend([(obj, type_mapping[obj]) for obj in objects])
            parametrized_predicate = predicate
            for obj in objects:
                parametrized_predicate = replace_whole_word(parametrized_predicate, obj, '?' + obj)
            effects[parametrized_predicate] = value

    parameters = list(set(parameters))  # remove duplicates

    return Action(parameters, preconditions, effects, name=name)

def operator_learner_minimal_effects_grounded(predicates_t, predicates_t1, type_mapping, name=None):
    parameters = []
    preconditions = {}
    effects = {}

    for predicate, value in predicates_t.items():
        objects = extract_objects_from_predicate(predicate)
        #parameters.extend([(obj, type_mapping[obj]) for obj in objects])
        parametrized_predicate = predicate
        #for obj in objects:
        #    parametrized_predicate = replace_whole_word(parametrized_predicate, obj, '?' + obj)
        preconditions[parametrized_predicate] = value

    for predicate, value in predicates_t1.items():
        if predicate not in predicates_t or predicates_t1[predicate] != predicates_t[predicate]:
            objects = extract_objects_from_predicate(predicate)
            parameters.extend([(obj, type_mapping[obj]) for obj in objects])
            parametrized_predicate = predicate
            #for obj in objects:
            #    parametrized_predicate = replace_whole_word(parametrized_predicate, obj, '?' + obj)
            effects[parametrized_predicate] = value

    parameters = list(set(parameters))  # remove duplicates

    return Action(parameters, preconditions, effects, name=name)

def operator_learner_nothing_grounded(predicates_t, predicates_t1, type_mapping, name=None):
    parameters = []
    preconditions = {}
    effects = {}

    for predicate, value in predicates_t.items():
        objects = extract_objects_from_predicate(predicate)
        parameters.extend([(obj, type_mapping[obj]) for obj in objects])
        parametrized_predicate = predicate
        for obj in objects:
            parametrized_predicate = replace_whole_word(parametrized_predicate, obj, '?' + obj)
        preconditions[parametrized_predicate] = value

    for predicate, value in predicates_t1.items():
        #if predicate not in predicates_t or predicates_t1[predicate] != predicates_t[predicate]:
        objects = extract_objects_from_predicate(predicate)
        parameters.extend([(obj, type_mapping[obj]) for obj in objects])
        parametrized_predicate = predicate
        for obj in objects:
            parametrized_predicate = replace_whole_word(parametrized_predicate, obj, '?' + obj)
        effects[parametrized_predicate] = value

    parameters = list(set(parameters))  # remove duplicates

    return Action(parameters, preconditions, effects, name=name)

def operator_learner_minimal_effects_grounded_preconditions(predicates_t, predicates_t1, type_mapping, name=None):
    parameters = []
    preconditions = {}
    effects = {}

    for predicate, value in predicates_t.items():
        objects = extract_objects_from_predicate(predicate)
        #parameters.extend([(obj, type_mapping[obj]) for obj in objects])
        parametrized_predicate = predicate
        #for obj in objects:
        #    parametrized_predicate = replace_whole_word(parametrized_predicate, obj, '?' + obj)
        preconditions[parametrized_predicate] = value

    for predicate, value in predicates_t1.items():
        if predicate not in predicates_t or predicates_t1[predicate] != predicates_t[predicate]:
            objects = extract_objects_from_predicate(predicate)
            parameters.extend([(obj, type_mapping[obj]) for obj in objects])
            parametrized_predicate = predicate
            for obj in objects:
                parametrized_predicate = replace_whole_word(parametrized_predicate, obj, '?' + obj)
            effects[parametrized_predicate] = value

    parameters = list(set(parameters))  # remove duplicates

    return Action(parameters, preconditions, effects, name=name)


def numerical_operator_learner(s1, s2, type_mapping, predicates_type, name=None, negatives=False):
    parameters = []
    preconditions = {}
    effects = {}
    numerical_preconditions = {}
    numerical_effects = {}
    function_effects = {}
    type_id = {type: 0 for type in type_mapping.values()}
    obj_id = {obj: None for obj in type_mapping.keys()}

    for predicate, value in s1.items():
        predicate_name = predicate.split('(')[0]
        objects = extract_objects_from_predicate(predicate)
        #parameters.extend([(obj, type_mapping[obj]) for obj in objects])
        parametrized_predicate = predicate
        for obj in objects:
            if obj_id[obj] is None:
                obj_id[obj] = type_mapping[obj] + str(type_id[type_mapping[obj]])
                type_id[type_mapping[obj]] += 1
            parameters.extend([(obj_id[obj], type_mapping[obj])])
            parametrized_predicate = replace_whole_word(parametrized_predicate, obj, '?' + obj_id[obj])
        if predicates_type[predicate_name] == 'num':
            numerical_preconditions[parametrized_predicate] = value
        elif predicates_type[predicate_name] == 'bool':
            #if value:
            preconditions[parametrized_predicate] = bool(value)
            #elif predicate in s2.items() and s2[predicate]:
            #    preconditions[parametrized_predicate] = bool(value)

    # If any bool predicate in the precondition, remove all numerical preconditions that do no evaluate to 0
    if any([predicates_type[predicate.split('(')[0]] == 'bool' for predicate in preconditions.keys()]):
        numerical_preconditions = {k: v for k, v in numerical_preconditions.items() if v <= 1}

    for predicate, value in s2.items():
        predicate_name = predicate.split('(')[0]
        if predicates_type[predicate_name] == 'num':
            if predicate in s1:
                objects = extract_objects_from_predicate(predicate)
                #parameters.extend([(obj, type_mapping[obj]) for obj in objects])
                parametrized_predicate = predicate
                for obj in objects:
                    if obj_id[obj] is None:
                        obj_id[obj] = type_mapping[obj] + str(type_id[type_mapping[obj]])
                        type_id[type_mapping[obj]] += 1
                    #parameters.extend([(obj_id[obj], type_mapping[obj])])
                    parametrized_predicate = replace_whole_word(parametrized_predicate, obj, '?' + obj_id[obj])
                    # Check if all objects in the predicate have been parametrized (with a ?)
                change = value - s1[predicate]
                if change != 0:
                    function_effects[parametrized_predicate] = change
            else:
                objects = extract_objects_from_predicate(predicate)
                #parameters.extend([(obj, type_mapping[obj]) for obj in objects])
                parametrized_predicate = predicate
                for obj in objects:
                    if obj_id[obj] is None:
                        obj_id[obj] = type_mapping[obj] + str(type_id[type_mapping[obj]])
                        type_id[type_mapping[obj]] += 1
                    #parameters.extend([(obj_id[obj], type_mapping[obj])])
                    parametrized_predicate = replace_whole_word(parametrized_predicate, obj, '?' + obj_id[obj])
                    numerical_effects[parametrized_predicate] = value
        elif predicates_type[predicate_name] == 'bool':
            if predicate not in s1 or s1[predicate] != value:
                objects = extract_objects_from_predicate(predicate)
                parametrized_predicate = predicate
                for obj in objects:
                    if obj_id[obj] is None:
                        obj_id[obj] = type_mapping[obj] + str(type_id[type_mapping[obj]])
                        type_id[type_mapping[obj]] += 1
                    #parameters.extend([(obj_id[obj], type_mapping[obj])])
                    parametrized_predicate = replace_whole_word(predicate, obj, '?' + obj_id[obj])
                effects[parametrized_predicate] = bool(value)

    parameters = list(set(parameters))  # remove duplicates

    return Action(parameters, preconditions, effects, numerical_preconditions, numerical_effects, function_effects, name=name)

def merge_actions(a0, a1):
    a0_function_effects = {k: v for k, v in a0.function_effects.items() if k != 'total-cost'}
    a1_function_effects = {k: v for k, v in a1.function_effects.items() if k != 'total-cost'}
    # Check if the actions are of the same type and have the same effects
    if a0.effects == a1.effects and a0.numerical_effects == a1.numerical_effects and a0_function_effects==a1_function_effects:
        # Merge the preconditions
        merging_a0_a1 = all(item in a1.preconditions.items() for item in a0.preconditions.items()) and all(item in a1.numerical_preconditions.items() for item in a0.numerical_preconditions.items())
        merging_a1_a0 = all(item in a0.preconditions.items() for item in a1.preconditions.items()) and all(item in a0.numerical_preconditions.items() for item in a1.numerical_preconditions.items())
        if merging_a0_a1:
            merged_preconditions = a0.preconditions
            merged_numerical_preconditions = a0.numerical_preconditions
        elif merging_a1_a0:
            merged_preconditions = a1.preconditions
            merged_numerical_preconditions = a1.numerical_preconditions
        else:
            # Remove precondition that are not in both actions or that have different values
            merged_preconditions = {k: v for k, v in a0.preconditions.items() if k in a1.preconditions and a1.preconditions[k] == v}
            merged_numerical_preconditions = {k: v for k, v in a0.numerical_preconditions.items() if k in a1.numerical_preconditions and a1.numerical_preconditions[k] == v}
            #merged_preconditions = {**a0.preconditions, **a1.preconditions}
            #merged_numerical_preconditions = {**a0.numerical_preconditions, **a1.numerical_preconditions}
        merged_action = Action(a0.parameters, merged_preconditions, a0.effects, merged_numerical_preconditions, a0.numerical_effects, a0.function_effects, name=a0.name)
        return merged_action
    else:
        return False

def split_action(action):
    # Split the action into a list of actions with one effect each
    # Create a list of actions to store the split actions 
    actions = []
    cost = action.function_effects['total-cost']
    counter = 0
    # Iterate over the function effects
    for effect, value in action.function_effects.items():
        if effect == 'total-cost':
            continue
        # Create a copy of the action
        new_action = Action(action.parameters.copy(), action.preconditions.copy(), {}, name=action.name + '_' + str(counter))
        # Add the effect to the new action
        new_action.function_effects = {effect: value, 'total-cost': cost}
        # Add the new action to the list of actions
        actions.append(new_action)
        counter += 1
    for effect, value in action.numerical_effects.items():
        # Create a copy of the action
        new_action = Action(action.parameters.copy(), action.preconditions.copy(), {}, function_effects={'total-cost':cost}, name=action.name + '_' + str(counter))
        # Add the effect to the new action
        new_action.numerical_effects = {effect: value}
        # Add the new action to the list of actions
        actions.append(new_action)
        counter += 1
    for effect, value in action.effects.items():
        # Create a copy of the action
        new_action = Action(action.parameters.copy(), action.preconditions.copy(), {}, function_effects={'total-cost':cost}, name=action.name + '_' + str(counter))
        # Add the effect to the new action
        new_action.effects = {effect: value}
        # Add the new action to the list of actions
        actions.append(new_action)
        counter += 1
    return actions

def update_action_cost(action, cost):
    if type(action) == list:
        for a in action:
            a.function_effects.update({'total-cost': cost})
    else:
        action.function_effects.update({'total-cost': cost})
    return action
