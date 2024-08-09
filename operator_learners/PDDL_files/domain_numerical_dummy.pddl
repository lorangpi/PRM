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

(:action a7_1
  :parameters (?door0 - door ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and (not (locked ?door0)) (increase (total-cost) 118) ))

(:action a23_0
  :parameters (?door0 - door ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (not (locked ?door0)) )
  :effect (and (not (grasped ?object0)) (increase (total-cost) 279) ))

(:action a116_1
  :parameters (?door0 - door ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and (grasped ?object0) (increase (total-cost) 177) ))

(:action a167_6
  :parameters (?door0 - door ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (open ?door0 ) 199) (increase (total-cost) 43) ))

(:action a176_1
  :parameters (?door0 - door ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (locked ?door0) )
  :effect (and (grasped ?object0) (increase (total-cost) 63) ))

(:action a179_0
  :parameters (?gripper0 - gripper ?door0 - door ?location3 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location3 ) 1) (increase (total-cost) 212) ))

(:action a250_1
  :parameters (?door0 - door ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (open ?door0 ) 1) (increase (total-cost) 11) ))

(:action a252_0
  :parameters (?gripper0 - gripper ?door0 - door ?location3 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (at_gripper ?gripper0 ?location3 ) 1) (increase (total-cost) 256) ))

(:action a267_0
  :parameters (?gripper0 - gripper ?door0 - door ?object0 - object ?location2 - location)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location2 ) 1) (increase (total-cost) 538) ))

(:action a269_0
  :parameters (?gripper0 - gripper ?door0 - door ?location3 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (locked ?door0) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location3 ) 1) (increase (total-cost) 138) ))

(:action a46_1
  :parameters (?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) )
  :effect (and (grasped ?object0) (increase (total-cost) 21) ))

(:action a300_0
  :parameters (?gripper0 - gripper ?door0 - door ?location3 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location3 ) 2) (increase (total-cost) 6) ))

(:action a301_1
  :parameters (?location1 - location ?door0 - door ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at ?object0 ?location1 ) 192) (increase (total-cost) 43) ))

(:action a301_2
  :parameters (?gripper0 - gripper ?door0 - door ?object0 - object ?location2 - location)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location2 ) 194) (increase (total-cost) 43) ))

(:action a301_5
  :parameters (?location1 - location ?gripper0 - gripper ?door0 - door ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location1 ) 192) (increase (total-cost) 43) ))

(:action a167_0
  :parameters (?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at ?object0 ?location0 ) 190) (increase (total-cost) 41) ))

(:action a312_3
  :parameters (?gripper0 - gripper ?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location0 ) 189) (increase (total-cost) 41) ))

(:action a313_0
  :parameters (?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at ?object0 ?location0 ) 188) (increase (total-cost) 43) ))

(:action a313_3
  :parameters (?gripper0 - gripper ?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location0 ) 188) (increase (total-cost) 43) ))

(:action a316_0
  :parameters (?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at ?object0 ?location0 ) 189) (increase (total-cost) 41) ))

(:action a167_1
  :parameters (?location1 - location ?door0 - door ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at ?object0 ?location1 ) 191) (increase (total-cost) 41) ))

(:action a316_4
  :parameters (?gripper0 - gripper ?door0 - door ?location3 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location3 ) 181) (increase (total-cost) 41) ))

(:action a167_5
  :parameters (?location1 - location ?gripper0 - gripper ?door0 - door ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location1 ) 191) (increase (total-cost) 41) ))

(:action a167_4
  :parameters (?gripper0 - gripper ?door0 - door ?location3 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location3 ) 182) (increase (total-cost) 41) ))

(:action a318_0
  :parameters (?location1 - location ?door0 - door ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at ?object0 ?location1 ) 1) (increase (total-cost) 39) ))

(:action a320_0
  :parameters (?gripper0 - gripper ?door0 - door ?location3 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location3 ) 1) (increase (total-cost) 58) ))

(:action a322_0
  :parameters (?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at ?object0 ?location0 ) 1) (increase (total-cost) 37) ))

(:action a167_2
  :parameters (?gripper0 - gripper ?door0 - door ?object0 - object ?location2 - location)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location2 ) 195) (increase (total-cost) 41) ))

(:action a307_4
  :parameters (?gripper0 - gripper ?door0 - door ?location3 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location3 ) 183) (increase (total-cost) 41) ))

(:action a325_0
  :parameters (?location1 - location ?door0 - door ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (decrease (at ?object0 ?location1 ) 1) (increase (total-cost) 41) ))

(:action a326_0
  :parameters (?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (decrease (at ?object0 ?location0 ) 1) (increase (total-cost) 41) ))

(:action a326_2
  :parameters (?gripper0 - gripper ?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location0 ) 1) (increase (total-cost) 64) ))

(:action a328_1
  :parameters (?location1 - location ?gripper0 - gripper ?door0 - door ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location1 ) 1) (increase (total-cost) 64) ))

(:action a330_0
  :parameters (?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at ?object0 ?location0 ) 188) (increase (total-cost) 41) ))

(:action a330_1
  :parameters (?location1 - location ?door0 - door ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at ?object0 ?location1 ) 192) (increase (total-cost) 42) ))

(:action a174_2
  :parameters (?gripper0 - gripper ?door0 - door ?object0 - object ?location2 - location)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location2 ) 193) (increase (total-cost) 41) ))

(:action a330_3
  :parameters (?gripper0 - gripper ?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location0 ) 188) (increase (total-cost) 41) ))

(:action a330_5
  :parameters (?location1 - location ?gripper0 - gripper ?door0 - door ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location1 ) 192) (increase (total-cost) 42) ))

(:action a334_0
  :parameters (?location1 - location ?gripper0 - gripper ?door0 - door ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (at_gripper ?gripper0 ?location1 ) 1) (increase (total-cost) 115) ))

(:action a335_0
  :parameters (?location1 - location ?door0 - door ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (at ?object0 ?location1 ) 1) (increase (total-cost) 115) ))

(:action a336_0
  :parameters (?gripper0 - gripper ?door0 - door ?object0 - object ?location2 - location)
  :precondition (and (grasped ?object0) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (at_gripper ?gripper0 ?location2 ) 1) (increase (total-cost) 193) ))

(:action a338_0
  :parameters (?gripper0 - gripper ?door0 - door ?location3 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (at_gripper ?gripper0 ?location3 ) 1) (increase (total-cost) 32) ))

(:action a339_0
  :parameters (?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (decrease (at ?object0 ?location0 ) 1) (increase (total-cost) 38) ))

(:action a340_0
  :parameters (?gripper0 - gripper ?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location0 ) 1) (increase (total-cost) 31) ))

(:action a341_0
  :parameters (?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (at ?object0 ?location0 ) 1) (increase (total-cost) 32) ))

(:action a342_0
  :parameters (?gripper0 - gripper ?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (at_gripper ?gripper0 ?location0 ) 1) (increase (total-cost) 87) ))

(:action a344_0
  :parameters (?location1 - location ?gripper0 - gripper ?door0 - door ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location1 ) 1) (increase (total-cost) 180) ))

(:action a345_0
  :parameters (?location1 - location ?door0 - door ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (decrease (at ?object0 ?location1 ) 1) (increase (total-cost) 50) ))

(:action a287_0
  :parameters (?location0 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) )
  :effect (and  (increase (at ?object0 ?location0 ) 1) (increase (total-cost) 24) ))

(:action a323_0
  :parameters (?gripper0 - gripper ?object0 - object ?location2 - location)
  :precondition (and (grasped ?object0) (not (light_off)) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location2 ) 1) (increase (total-cost) 27) ))

(:action a347_0
  :parameters (?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at ?object0 ?location0 ) 190) (increase (total-cost) 42) ))

(:action a347_2
  :parameters (?gripper0 - gripper ?door0 - door ?object0 - object ?location2 - location)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location2 ) 195) (increase (total-cost) 42) ))

(:action a347_3
  :parameters (?gripper0 - gripper ?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location0 ) 190) (increase (total-cost) 42) ))

(:action a347_4
  :parameters (?gripper0 - gripper ?door0 - door ?location3 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location3 ) 183) (increase (total-cost) 42) ))

(:action a353_0
  :parameters (?gripper0 - gripper ?door0 - door ?object0 - object ?location2 - location)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location2 ) 1) (increase (total-cost) 211) ))

(:action a353_1
  :parameters (?location1 - location ?gripper0 - gripper ?door0 - door ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (locked ?door0) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location1 ) 1) (increase (total-cost) 70) ))

(:action a355_0
  :parameters (?gripper0 - gripper ?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location0 ) 1) (increase (total-cost) 269) ))

(:action a356_0
  :parameters (?gripper0 - gripper ?door0 - door ?location3 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location3 ) 1) (increase (total-cost) 191) ))

(:action a357_0
  :parameters (?gripper0 - gripper ?door0 - door ?object0 - object ?location2 - location)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (locked ?door0) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location2 ) 1) (increase (total-cost) 79) ))

(:action a358_1
  :parameters (?gripper0 - gripper ?door0 - door ?location3 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (locked ?door0) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location3 ) 1) (increase (total-cost) 152) ))

(:action a360_0
  :parameters (?location1 - location ?gripper0 - gripper ?door0 - door ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location1 ) 1) (increase (total-cost) 213) ))

(:action a361_0
  :parameters (?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (locked ?door0) )
  :effect (and  (decrease (at ?object0 ?location0 ) 1) (increase (total-cost) 28) ))

(:action a362_1
  :parameters (?location1 - location ?gripper0 - gripper ?door0 - door ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location1 ) 1) (increase (total-cost) 248) ))

(:action a371_0
  :parameters (?location1 - location ?door0 - door ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at ?object0 ?location1 ) 1) (increase (total-cost) 23) ))

(:action a372_0
  :parameters (?gripper0 - gripper ?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (locked ?door0) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location0 ) 1) (increase (total-cost) 63) ))

(:action a374_0
  :parameters (?gripper0 - gripper ?door0 - door ?location3 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (at_gripper ?gripper0 ?location3 ) 1) (increase (total-cost) 79) ))

(:action a375_0
  :parameters (?gripper0 - gripper ?door0 - door ?object0 - object ?location2 - location)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location2 ) 1) (increase (total-cost) 64) ))

(:action a377_1
  :parameters (?location1 - location ?door0 - door ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (locked ?door0) )
  :effect (and  (decrease (at ?object0 ?location1 ) 2) (increase (total-cost) 11) ))

(:action a378_1
  :parameters (?location1 - location ?door0 - door ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (locked ?door0) )
  :effect (and  (decrease (at ?object0 ?location1 ) 3) (increase (total-cost) 11) ))

(:action a379_0
  :parameters (?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at ?object0 ?location0 ) 3) (increase (total-cost) 6) ))

(:action a211_0
  :parameters (?gripper0 - gripper ?location3 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) )
  :effect (and  (increase (at_gripper ?gripper0 ?location3 ) 1) (increase (total-cost) 19) ))

(:action a188_0
  :parameters (?gripper0 - gripper ?door0 - door ?object0 - object ?location2 - location)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (locked ?door0) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location2 ) 1) (increase (total-cost) 28) ))

(:action a381_0
  :parameters (?gripper0 - gripper ?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (at_gripper ?gripper0 ?location0 ) 1) (increase (total-cost) 75) ))

(:action a382_0
  :parameters (?gripper0 - gripper ?door0 - door ?object0 - object ?location2 - location)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (at_gripper ?gripper0 ?location2 ) 1) (increase (total-cost) 253) ))

(:action a383_0
  :parameters (?gripper0 - gripper ?door0 - door ?location3 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location3 ) 1) (increase (total-cost) 334) ))

(:action a384_0
  :parameters (?gripper0 - gripper ?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location0 ) 1) (increase (total-cost) 51) ))

(:action a385_0
  :parameters (?door0 - door ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (decrease (open ?door0 ) 14) (increase (total-cost) 26) ))

(:action a386_1
  :parameters (?door0 - door ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (open ?door0 ) 2) (increase (total-cost) 26) ))

(:action a308_0
  :parameters (?door0 - door)
  :precondition (and (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (open ?door0 ) 2) (increase (total-cost) 16) ))

(:action a387_0
  :parameters (?location1 - location ?door0 - door ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (at ?object0 ?location1 ) 1) (increase (total-cost) 219) ))

(:action a318_1
  :parameters (?location1 - location ?gripper0 - gripper ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) )
  :effect (and  (increase (at_gripper ?gripper0 ?location1 ) 1) (increase (total-cost) 27) ))

(:action a388_4
  :parameters (?location1 - location ?gripper0 - gripper ?door0 - door ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location1 ) 1) (increase (total-cost) 85) ))

(:action a195_0
  :parameters (?door0 - door)
  :precondition (and (not (light_off)) (not (locked ?door0)) )
  :effect (and  (decrease (open ?door0 ) 14) (increase (total-cost) 16) ))

(:action a389_0
  :parameters (?gripper0 - gripper ?door0 - door ?object0 - object ?location2 - location)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location2 ) 1) (increase (total-cost) 180) ))

(:action a391_0
  :parameters (?location1 - location ?door0 - door ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (decrease (at ?object0 ?location1 ) 1) (increase (total-cost) 28) ))

(:action a367_0
  :parameters (?gripper0 - gripper ?location0 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location0 ) 1) (increase (total-cost) 84) ))

(:action a281_0
  :parameters (?location1 - location ?object0 - object)
  :precondition (and (not (light_off)) )
  :effect (and  (decrease (at ?object0 ?location1 ) 1) (increase (total-cost) 26) ))

(:action a392_0
  :parameters (?gripper0 - gripper ?door0 - door ?location3 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location3 ) 1) (increase (total-cost) 64) ))

(:action a310_0
  :parameters (?gripper0 - gripper ?door0 - door ?location0 - location)
  :precondition (and (not (light_off)) (not (locked ?door0)) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location0 ) 1) (increase (total-cost) 7) ))

(:action a346_0
  :parameters (?gripper0 - gripper ?location3 - location)
  :precondition (and (not (light_off)) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location3 ) 1) (increase (total-cost) 32) ))

(:action a393_0
  :parameters (?door0 - door ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (decrease (open ?door0 ) 14) (increase (total-cost) 6) ))

(:action a394_0
  :parameters (?location1 - location ?gripper0 - gripper ?door0 - door ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (at_gripper ?gripper0 ?location1 ) 1) (increase (total-cost) 109) ))

(:action a395_0
  :parameters (?gripper0 - gripper ?door0 - door ?object0 - object ?location2 - location)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location2 ) 1) (increase (total-cost) 41) ))

(:action a396_0
  :parameters (?gripper0 - gripper ?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location0 ) 1) (increase (total-cost) 37) ))

(:action a376_0
  :parameters (?gripper0 - gripper ?location0 - location)
  :precondition (and (not (light_off)) )
  :effect (and  (increase (at_gripper ?gripper0 ?location0 ) 1) (increase (total-cost) 33) ))

(:action a398_2
  :parameters (?gripper0 - gripper ?door0 - door ?location3 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location3 ) 1) (increase (total-cost) 162) ))

(:action a283_1
  :parameters (?location1 - location ?gripper0 - gripper ?door0 - door ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location1 ) 1) (increase (total-cost) 27) ))

(:action a274_0
  :parameters (?location0 - location ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) )
  :effect (and  (decrease (at ?object0 ?location0 ) 1) (increase (total-cost) 26) ))

(:action a368_1
  :parameters (?gripper0 - gripper ?location2 - location)
  :precondition (and (not (light_off)) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location2 ) 1) (increase (total-cost) 24) ))

(:action a282_0
  :parameters (?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at ?object0 ?location0 ) 1) (increase (total-cost) 31) ))

(:action a399_1
  :parameters (?location1 - location ?door0 - door ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at ?object0 ?location1 ) 2) (increase (total-cost) 12) ))

(:action a401_1
  :parameters (?location1 - location ?door0 - door ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at ?object0 ?location1 ) 4) (increase (total-cost) 11) ))

(:action a402_0
  :parameters (?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (decrease (at ?object0 ?location0 ) 1) (increase (total-cost) 37) ))

(:action a226_0
  :parameters (?location0 - location ?object0 - object)
  :precondition (and (not (light_off)) )
  :effect (and  (increase (at ?object0 ?location0 ) 1) (increase (total-cost) 11) ))

(:action a403_0
  :parameters (?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (at ?object0 ?location0 ) 1) (increase (total-cost) 25) ))

(:action a404_4
  :parameters (?door0 - door ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (open ?door0 ) 27) (increase (total-cost) 31) ))

(:action a405_0
  :parameters (?door0 - door ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (decrease (open ?door0 ) 4) (increase (total-cost) 31) ))

(:action a406_3
  :parameters (?door0 - door ?object0 - object)
  :precondition (and (grasped ?object0) (not (light_off)) (locked ?door0) )
  :effect (and  (increase (open ?door0 ) 27) (increase (total-cost) 26) ))

(:action a194_0
  :parameters (?door0 - door)
  :precondition (and (not (light_off)) (locked ?door0) )
  :effect (and  (increase (open ?door0 ) 27) (increase (total-cost) 21) ))

(:action a407_0
  :parameters (?gripper0 - gripper ?door0 - door ?object0 - object ?location2 - location)
  :precondition (and (grasped ?object0) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location2 ) 1) (increase (total-cost) 29) ))

(:action a408_0
  :parameters (?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (at ?object0 ?location0 ) 196) (increase (total-cost) 41) ))

(:action a408_1
  :parameters (?location1 - location ?door0 - door ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (at ?object0 ?location1 ) 189) (increase (total-cost) 41) ))

(:action a408_2
  :parameters (?gripper0 - gripper ?door0 - door ?object0 - object ?location2 - location)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (at_gripper ?gripper0 ?location2 ) 192) (increase (total-cost) 41) ))

(:action a167_3
  :parameters (?gripper0 - gripper ?location0 - location)
  :precondition (and (not (light_off)) )
  :effect (and  (increase (at_gripper ?gripper0 ?location0 ) 190) (increase (total-cost) 41) ))

(:action a174_4
  :parameters (?gripper0 - gripper ?location3 - location)
  :precondition (and (not (light_off)) )
  :effect (and  (increase (at_gripper ?gripper0 ?location3 ) 184) (increase (total-cost) 41) ))

(:action a408_5
  :parameters (?location1 - location ?gripper0 - gripper ?door0 - door ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (at_gripper ?gripper0 ?location1 ) 190) (increase (total-cost) 41) ))

(:action a408_6
  :parameters (?door0 - door ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (open ?door0 ) 186) (increase (total-cost) 41) ))

(:action a284_1
  :parameters (?door0 - door)
  :precondition (and (not (light_off)) (not (locked ?door0)) )
  :effect (and  (decrease (open ?door0 ) 4) (increase (total-cost) 11) ))

(:action a364_0
  :parameters (?gripper0 - gripper ?location2 - location)
  :precondition (and (not (light_off)) )
  :effect (and  (decrease (at_gripper ?gripper0 ?location2 ) 1) (increase (total-cost) 17) ))

(:action a351_0
  :parameters (?location0 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) )
  :effect (and  (increase (at ?object0 ?location0 ) 1) (increase (total-cost) 26) ))

(:action a351_1
  :parameters (?location1 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) )
  :effect (and  (decrease (at ?object0 ?location1 ) 1) (increase (total-cost) 55) ))

(:action a278_0
  :parameters (?location1 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) )
  :effect (and  (increase (at ?object0 ?location1 ) 3) (increase (total-cost) 16) ))

(:action a413_0
  :parameters (?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (at ?object0 ?location0 ) 3) (increase (total-cost) 16) ))

(:action a414_0
  :parameters (?door0 - door ?location0 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (not (locked ?door0)) )
  :effect (and  (increase (at ?object0 ?location0 ) 2) (increase (total-cost) 16) ))

(:action a390_1
  :parameters (?location1 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) )
  :effect (and  (increase (at ?object0 ?location1 ) 2) (increase (total-cost) 16) ))

(:action a278_0
  :parameters (?location0 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) )
  :effect (and  (increase (at ?object0 ?location0 ) 3) (increase (total-cost) 11) ))

(:action a192_0
  :parameters (?location0 - location ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) )
  :effect (and  (increase (at ?object0 ?location0 ) 4) (increase (total-cost) 11) ))

(:action a324_1
  :parameters (?gripper0 - gripper ?door0 - door ?location0 - location)
  :precondition (and (not (light_off)) (locked ?door0) )
  :effect (and  (increase (at_gripper ?gripper0 ?location0 ) 1) (increase (total-cost) 20) ))

(:action a415_0
  :parameters (?location1 - location ?door0 - door ?object0 - object)
  :precondition (and (not (grasped ?object0)) (not (light_off)) (locked ?door0) )
  :effect (and  (decrease (at ?object0 ?location1 ) 1) (increase (total-cost) 25) ))
)