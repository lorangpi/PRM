(define (domain gripper-typed)

(:requirements
  :typing
  :strips)

(:types location object gripper)

(:constants left right - gripper)

(:predicates (at-robby ?r - location) (at ?b - object ?r - location) (free ?g - gripper) (carry ?o - object ?g - gripper))

(:action move
  :parameters (?from - location ?to - location)
  :precondition (at-robby ?from)
  :effect (and (at-robby ?to) (not (at-robby ?from))))

(:action pick
  :parameters (?obj - object ?location - location ?gripper - gripper)
  :precondition (and (at ?obj ?location) (at-robby ?location) (free ?gripper))
  :effect (and (carry ?obj ?gripper) (not (at ?obj ?location)) (not (free ?gripper))))

(:action drop
  :parameters (?obj - object ?location - location ?gripper - gripper)
  :precondition (and (carry ?obj ?gripper) (at-robby ?location))
  :effect (and (at ?obj ?location) (free ?gripper) (not (carry ?obj ?gripper))))

)