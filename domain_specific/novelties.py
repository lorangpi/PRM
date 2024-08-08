from robosuite import Hole, Locked, Elevated, Lightoff, Obstacle

# Mapping from novelty name to a setup dictionary. "type" is assumed to be detected, the novelty should be charaterized as global or local, in which
# case it changes the priority of the source policy to trasnfer from when accomodating to novelties.

novelties_info = {
    "Hole": {"wrapper":Hole, "params": None, "type":"local", "pattern": "Hole"},
    "Locked": {"wrapper":Locked, "params": None, "type":"local", "pattern": "Locked"},
    "Elevated": {"wrapper":Elevated, "params": None, "type":"local", "pattern": "Elevated"},
    "Lightoff": {"wrapper":Lightoff, "params": None, "type":"local", "pattern": "Lightoff"},
    "Obstacle": {"wrapper":Obstacle, "params": None, "type":"local", "pattern": "Obstacle"}
}