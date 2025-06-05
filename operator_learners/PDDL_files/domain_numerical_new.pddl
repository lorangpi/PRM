(define (domain generatedDomain)
  (:requirements :typing)
  (:types object location gripper)

(:constants left right - gripper)

(:predicates
  (grasped ?o - object)
  (door_collision)
  (dropped_off)
  (light_off)
  (locked ?o - object)
  (picked_up ?o - object)
)

(:functions
  (total-cost) - number
  (at_gripper ?g - gripper ?l - location) - number
  (at ?o - object ?l - location) - number
  (at_grab_level ?g - gripper ?o - object) - number
  (open ?o - object) - number
  (over ?g - gripper ?o - object) - number
)

)