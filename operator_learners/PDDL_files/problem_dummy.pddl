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
	(= (at can drop ) 8)
	(= (at can pick ) 4)
	(= (at_gripper gripper activate ) 5)
	(= (at_gripper gripper drop ) 8)
	(= (at_gripper gripper lightswitch ) 14)
	(= (at_gripper gripper pick ) 4)
	(grasped can)
	(locked door)
	(= (open door ) 0)
	(= (open gripper ) 0)
	(= (total-cost) 0)

	)
(:goal (and (= (at can drop ) 1)) 
)
(:metric minimize (total-cost))


)