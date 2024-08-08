(define
    (problem adeGeneratedProblem)
    (:domain gripper-typed)
(:objects
        rooma - room
        roomb - room
        b1 - ball
        b2 - ball
    )
(:init
    (at-robby roomb )
    (free left )
    (at b1 rooma )
    (at b2 rooma )
)
(:goal
    (and 
        (at b1 roomb)
        (at b2 roomb)
    )
)
)