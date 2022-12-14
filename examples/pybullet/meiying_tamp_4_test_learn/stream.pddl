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
  (:stream sample-pull-tool-goal
    :inputs (?o ?to ?r)
    :domain (and (Stackable ?o ?r) (Floor ?r) (PullTool ?to))
    :outputs (?p)
    :certified (and (PullToolGoalPose ?o ?p ?to) (Pose ?o ?p) (Supported ?o ?p ?r))
  )
  (:stream sample-push-tool-goal
    :inputs (?o ?to ?r)
    :domain (and (Stackable ?o ?r) (Floor ?r) (PushTool ?to))
    :outputs (?p)
    :certified (and (PushToolGoalPose ?o ?p ?to) (Pose ?o ?p) (Supported ?o ?p ?r))
  )
  (:stream inverse-kinematics
    :inputs (?o ?p1 ?g)
    :domain (and (Pose ?o ?p1) (Grasp ?o ?g))
    :outputs (?p2 ?q2 ?t)
    :certified (and (Conf ?q2) (Traj ?t) (Kin ?o ?p1 ?p2 ?g ?q2 ?t) (ValidGrasp ?o ?p2 ?q2))
  )
  (:stream plan-pull-ik
    :inputs (?o ?to ?p1 ?p2 ?g ?r)
    :domain (and (Pose ?o ?p1) (Floor ?r) (PullToolGoalPose ?o ?p2 ?to) (PullTool ?to) (Different ?o ?to) (Manipulanda ?o) (Grasp ?to ?g))
    :outputs (?p3 ?top1 ?top ?q1 ?q2 ?t)
    :certified (and (PullConfigKin ?o ?p1 ?p2 ?p3 ?to ?top1 ?g ?top ?q1 ?q2 ?t) 
                    (PullConfig ?o ?p1 ?p3 ?to ?g)
                    (ValidGrasp ?to ?top1 ?q1) 
                    (Conf ?q1) 
                    (Conf ?q2) 
                    (Traj ?t) 
                    (Pose ?o ?p3)
                    (Supported ?o ?p3 ?r)
               )
  )
  (:stream plan-push-ik
    :inputs (?o ?to ?p1 ?p2 ?g ?r)
    :domain (and (Pose ?o ?p1) (Floor ?r) (PushToolGoalPose ?o ?p2 ?to) (PushTool ?to) (Different ?o ?to) (Manipulanda ?o) (Grasp ?to ?g))
    :outputs (?p3 ?top1 ?top ?q1 ?q2 ?t)
    :certified (and (PushConfigKin ?o ?p1 ?p2 ?p3 ?to ?top1 ?g ?top ?q1 ?q2 ?t) 
                    (PushConfig ?o ?p1 ?p3 ?to ?g)
                    (ValidGrasp ?to ?top1 ?q1) 
                    (Conf ?q1) 
                    (Conf ?q2) 
                    (Traj ?t) 
                    (Pose ?o ?p3)
                    (Supported ?o ?p3 ?r)
               )
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
  (:stream test-cfree-pull-pose
    :inputs (?o ?p1 ?p2 ?to ?o2 ?p22 ?g)
    :domain (and (Pose ?o ?p1) (Pose ?o ?p2) (Pose ?o2 ?p22)
                 (PullTool ?to) (Movable ?o) (Movable ?o2) (Different ?o ?to) (Different ?o2 ?to) (Different ?o2 ?o)
                 (PullConfig ?o ?p1 ?p2 ?to ?g)
            )
    :certified (CFreePullPose ?o ?p1 ?p2 ?to ?g ?o2 ?p22)
  )
  (:stream test-cfree-pull-traj-pose
    :inputs (?o ?to ?t ?o2 ?p2)
    :domain (and (PullTool ?to) (Traj ?t) (Movable ?o) (Pose ?o2 ?p2) (Different ?o ?to) (Different ?o ?o2) (Different ?o2 ?to))
    :certified (CFreePullTrajPose ?o ?to ?t ?o2 ?p2)
  )
  (:stream test-cfree-push-pose
    :inputs (?o ?p1 ?p2 ?to ?o2 ?p22 ?g)
    :domain (and (Pose ?o ?p1) (Pose ?o ?p2) (Pose ?o2 ?p22)
                 (PushTool ?to) (Movable ?o) (Movable ?o2) (Different ?o ?to) (Different ?o2 ?to) (Different ?o2 ?o)
                 (PushConfig ?o ?p1 ?p2 ?to ?g)
            )
    :certified (CFreePushPose ?o ?p1 ?p2 ?to ?g ?o2 ?p22)
  )
  (:stream test-cfree-push-traj-pose
    :inputs (?o ?to ?t ?o2 ?p2)
    :domain (and (PushTool ?to) (Traj ?t) (Movable ?o) (Pose ?o2 ?p2) (Different ?o ?to) (Different ?o ?o2) (Different ?o2 ?to))
    :certified (CFreePushTrajPose ?o ?to ?t ?o2 ?p2)
  )
  (:stream test-pick-feasible
    :inputs (?o ?r ?f ?p)
    :domain (and (Pose ?o ?p) (Floor ?f) (Supported ?o ?p ?r) (Region ?r))
    :certified (PickFeasible ?o ?r ?f ?p)
  )
  (:stream test-place-feasible
    :inputs (?o ?s1 ?r)
    :domain (and (Stackable ?o ?r) (Region ?s1))
    :certified (PlaceFeasible ?o ?s1 ?r)
  )
  (:stream test-move-feasible
    :inputs (?o1 ?p11 ?p12 ?q11 ?q12 ?o2 ?p21)
    :domain (and (Pose ?o2 ?p21) (ValidGrasp ?o1 ?p11 ?q11) (ValidGrasp ?o1 ?p12 ?q12))
    :certified (MoveFeasible ?o1 ?p11 ?p12 ?q11 ?q12 ?o2 ?p21)
  )
  (:stream test-pull-feasible
    :inputs (?o ?to ?p)
    :domain (and (Pose ?o ?p) (PullTool ?to) (Different ?o ?to) (Manipulanda ?o))
    :certified (PullFeasible ?o ?to ?p)
  )
  (:stream test-push-feasible
    :inputs (?o ?to ?p)
    :domain (and (Pose ?o ?p) (PushTool ?to) (Different ?o ?to) (Manipulanda ?o))
    :certified (PushFeasible ?o ?to ?p)
  )
)
