(define (domain generatedDomain)

(:requirements
  :typing
  :strips)

(:types location object gripper)

(:constants left right - gripper)

(:predicates (at_gripper ?g - gripper ?l - location) (at ?b - object ?r - location) (grasped ?o - object) (picked_up ?o - object) 
(at_grab_level ?g - gripper ?o - object) (door_collision) (dropped_off) (light_off) (locked  ?o - object) (open ?o - object) (over  ?g - gripper  ?o - object)
)

(:action move
  :parameters (?from - location ?to - location ?obj - object)
  :precondition (at ?obj ?from)
  :effect (and (at ?obj ?to) (not (at ?obj ?from))))

(:action pick
  :parameters (?obj - object ?location - location ?gripper - gripper)
  :precondition (and (at ?obj ?location) (at_gripper ?gripper ?location) (not (picked_up ?obj )))
  :effect (and (picked_up ?obj ) (not (at ?obj ?location))))

(:action drop
  :parameters (?obj - object ?location - location ?gripper - gripper)
  :precondition (and (picked_up ?obj ) (at_gripper ?gripper ?location))
  :effect (and (at ?obj ?location) (not (picked_up ?obj ))))

)