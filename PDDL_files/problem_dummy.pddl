(define (problem generatedProblem)
	(:domain generatedDomain)
(:objects
		can - object
		door - object
		pick - location
		drop - location
		gripper - gripper
	)
(:init
	(at can pick)
	(at_gripper gripper pick)
	(locked door)
	(open gripper)
	(= (total-cost) 0)

	)
(:goal (and (at can drop)(not (picked_up can ))) 
)
(:metric minimize (total-cost))


)