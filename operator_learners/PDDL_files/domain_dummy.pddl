(define (domain generatedDomain)

(:requirements
  :typing
  :strips)

(:types location object gripper)

(:constants left right - gripper)

(:predicates (at_gripper ?g - gripper ?l - location) (at ?b - object ?r - location) (grasped ?o - object) (picked_up ?o - object) 
(at_grab_level ?g - gripper ?o - object) (door_collision) (dropped_off) (light_off) (locked  ?o - object) (open ?o - object) (over  ?g - gripper  ?o - object)
)

(:action move0
  :parameters (?from - location ?to - location ?obj - object)
  :precondition (and (at ?obj ?from) (not (= ?from pick)) )
  :effect (and (at ?obj ?to) (not (at ?obj ?from)) (increase (total-cost) 1000) ))

(:action move1
  :parameters (?from - location ?to - location ?obj - object)
  :precondition (and (at ?obj ?from) (not (= ?to drop)) )
  :effect (and (at ?obj ?to) (not (at ?obj ?from)) (increase (total-cost) 1000) ))

(:action move2
  :parameters (?from - location ?to - location ?obj - object)
  :precondition (and (at ?obj ?from) (not (= ?obj gripper)) )
  :effect (and (at ?obj ?to) (not (at ?obj ?from)) (increase (total-cost) 1000) ))

(:action pick
  :parameters (?obj - object ?location - location ?gripper - gripper)
  :precondition (and (at_gripper ?gripper ?location) (at ?obj ?location) (not (picked_up ?obj)) )
  :effect (and (picked_up ?obj) (not (at ?obj ?location)) (increase (total-cost) 1000) ))

(:action drop
  :parameters (?obj - object ?location - location ?gripper - gripper)
  :precondition (and (at_gripper ?gripper ?location) (picked_up ?obj) )
  :effect (and (at ?obj ?location) (not (picked_up ?obj)) (increase (total-cost) 1000) ))
)