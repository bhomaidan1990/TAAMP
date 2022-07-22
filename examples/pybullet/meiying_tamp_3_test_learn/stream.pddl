(define (stream kuka-tamp)
  (:stream sample-pose
    :inputs (?o ?r)
    :domain (and (Stackable ?o ?r))
    :outputs (?p)
    :certified (and (Pose ?o ?p) (Supported ?o ?p ?r))
  )
  (:stream sample-grasp
    :inputs (?o)
    :domain (Graspable ?o)
    :outputs (?g)
    :certified (Grasp ?o ?g)
  )
  (:stream inverse-kinematics
    :inputs (?o ?p1 ?g)
    :domain (and (Pose ?o ?p1) (Grasp ?o ?g))
    :outputs (?p2 ?q2 ?t)
    :certified (and (Conf ?q2) (Traj ?t) (Kin ?o ?p1 ?p2 ?g ?q2 ?t) (ValidGrasp ?o ?p2 ?q2))
  )
  (:stream plan-free-motion
    :inputs (?q1 ?q2)
    :domain (and (Conf ?q1) (Conf ?q2))
    :fluents (AtPose) ; AtGrasp
    :outputs (?t)
    ;:certified (and (Traj ?t) (FreeMotion ?q1 ?t ?q2))
    :certified (FreeMotion ?q1 ?t ?q2)
  )
  (:stream plan-holding-motion
    :inputs (?q1 ?q2 ?o ?g ?p1 ?p2)
    :domain (and (Conf ?q1) (Conf ?q2) (Grasp ?o ?g) (ValidGrasp ?o ?p1 ?q1) (ValidGrasp ?o ?p2 ?q2))
    :fluents (AtPose)
    :outputs (?t)
    ;:certified (and (Traj ?t) (HoldingMotion ?q1 ?t ?q2 ?o ?g))
    :certified (HoldingMotion ?q1 ?t ?q2 ?o ?p1 ?p2 ?g)
  )

  (:stream test-cfree-pose-pose
    :inputs (?o1 ?p1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Pose ?o2 ?p2))
    :certified (CFreePosePose ?o1 ?p1 ?o2 ?p2)
  )
  (:stream test-cfree-approach-pose
    :inputs (?o1 ?p1 ?g1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Grasp ?o1 ?g1) (Pose ?o2 ?p2))
    :certified (CFreeApproachPose ?o1 ?p1 ?g1 ?o2 ?p2)
  )
  (:stream test-cfree-traj-pose
    :inputs (?t ?o2 ?p2)
    :domain (and (Traj ?t) (Pose ?o2 ?p2))
    :certified (CFreeTrajPose ?t ?o2 ?p2)
  )
  (:stream test-pick-feasible
    :inputs (?o ?r ?f ?p)
    :domain (and (Pose ?o ?p) (Floor ?f) (Supported ?o ?p ?r) (Region ?r))
    :certified (PickFeasible ?o ?r ?f ?p)
  )  
  (:stream test-place-feasible
    :inputs (?o ?s ?r)
    :domain (and (Stackable ?o ?r) (Region ?s))
    :certified (PlaceFeasible ?o ?s ?r)
  )
)
