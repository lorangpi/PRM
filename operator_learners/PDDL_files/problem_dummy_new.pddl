(define (problem generatedProblem)
	(:domain generatedDomain)
(:objects
		can - object
		door - door
		pick - location
		drop - location
		activate - location
		lightswitch - location
		gripper - gripper
	)
(:init
	(= (at can drop ) 10)
	(= (at can pick ) 2)
	(= (at_gripper gripper activate ) 5)
	(= (at_gripper gripper drop ) 10)
	(= (at_gripper gripper lightswitch ) 15)
	(= (at_gripper gripper pick ) 2)
	(grasped can)
	(locked door)
	(= (open door ) 0)
	(= (open gripper ) 0)
	(= (total-cost) 0)

	)
(:goal (and (= (at can drop ) 1)(grasped can)) 
)
(:metric minimize (total-cost))


)