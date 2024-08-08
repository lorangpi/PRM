(define (problem generatedProblem)
	(:domain generatedDomain)
(:objects
		can - object
		door - object
		pick - location
		drop - location
		grip - gripper
	)
(:init
	(at can pick)
	(at_gripper grip pick)
	(over grip can)
	(picked_up can)

	)
(:goal (and (at can drop))) 
)

