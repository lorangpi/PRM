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
	(open gripper)

	)
(:goal (and (at can drop)(not (picked_up can )))) 
)

