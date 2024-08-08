(define (problem generatedProblem)
	(:domain generatedDomain)
(:objects
		can door - object
		pick drop - location
		grip - gripper
	)
(:init
	(= (at can pick) 0)
	(= (at_gripper grip pick) 5)
	;(= (over grip can) 0)
	(picked_up can)

	)
(:goal (and (= (at can drop) 0))) 

(:metric minimize (total-cost))

)

