(define (stream kuka-tamp)
  (:stream sample-pose
    :inputs (?o ?r)
    :domain (Stackable ?o ?r)
    :outputs (?p)
    :certified (and (Pose ?o ?p) (Supported ?o ?p ?r))
  )
  (:stream sample-grasp
    :inputs (?o)
    :domain (Graspable ?o)
    :outputs (?g)
    :certified (Grasp ?o ?g)
  )
  (:stream sample-tool-goal
    :inputs (?o ?r)
    :domain (and (Stackable ?o ?r) (Floor ?r))
    :outputs (?p)
    :certified (and (ToolGoalPose ?o ?p) (Pose ?o ?p) (Supported ?o ?p ?r))
  )
  (:stream inverse-kinematics
    :inputs (?o ?p ?g)
    :domain (and (Pose ?o ?p) (Grasp ?o ?g))
    :outputs (?q ?t)
    :certified (and (Conf ?q) (Traj ?t) (Kin ?o ?p ?g ?q ?t))
  )
  ;(:stream pull-inverse-kinematics
  ;  :inputs (?o ?p1 ?p2 ?to ?top1 ?top2 ?cg ?g)
  ;  :domain (and (Pose ?o ?p1) (Pose ?o ?p2) (Tool ?to) (Different ?o ?to) (Movable ?o)
  ;               (PullConfig ?o ?p1 ?p2 ?to ?top1 ?top2 ?cg ?g)
  ;               ;(PullConfig ?o ?p1 ?p2 ?to ?g ?cg)
  ;               (Grasp ?to ?g))
  ;  :outputs (?q ?top ?t)
  ;  :certified (and (PullKin ?to ?top1 ?top2 ?top ?g ?q ?t) (Pose ?to ?top) (Traj ?t))
  ;)
  ;(:stream pull-inverse-kinematics
  ;  :inputs (?o ?p1 ?p2 ?to ?g)
  ;  :domain (and ;(Pose ?o ?p1)
  ;               ;(Pose ?o ?p2)
  ;               (Tool ?to)
  ;               (Different ?o ?to)
  ;               (Movable ?o)
  ;               (Grasp ?to ?g)
  ;               ;(PullConfig ?o ?p1 ?p2 ?to ?top1 ?top2 ?cg ?g)
  ;          )
  ;  :outputs (?top ?t)
  ;  :certified (and (PullKinTemp ?top ?t)
  ;                  ;(Pose ?to ?top)
  ;                  ;(Traj ?t)
  ;             )
  ;)
  (:stream plan-pull-ik
    :inputs (?o ?to ?p1 ?p2 ?g)
    :domain (and (Pose ?o ?p1) (ToolGoalPose ?o ?p2) (Tool ?to) (Different ?o ?to) (Manipulanda ?o) (Grasp ?to ?g))
    ;:outputs (?top1 ?top2 ?cg)
    ;:certified (and (PullConfig ?o ?p1 ?p2 ?to ?top1 ?top2 ?cg ?g))
    :outputs (?p3 ?top ?q1 ?q2 ?t)
    ;:certified (and (PullConfigTemp ?o ?p1 ?p2 ?to ?top1 ?top2 ?cg ?g))
    ;:certified (and (PullConfig ?o ?p1 ?p2 ?to ?top1 ?top2 ?cg ?g))
    :certified (and (PullConfigKin ?o ?p1 ?p2 ?p3 ?to ?g ?top ?q1 ?q2 ?t) (PullConfig ?o ?p1 ?p3 ?to ?g) (Conf ?q1) (Conf ?q2) (Traj ?t) (Pose ?o ?p3))
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
    :inputs (?q1 ?q2 ?o ?g)
    :domain (and (Conf ?q1) (Conf ?q2) (Grasp ?o ?g))
    :fluents (AtPose)
    :outputs (?t)
    ;:certified (and (Traj ?t) (HoldingMotion ?q1 ?t ?q2 ?o ?g))
    :certified (HoldingMotion ?q1 ?t ?q2 ?o ?g)
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
                 (Tool ?to) (Movable ?o) (Movable ?o2) (Different ?o ?to) (Different ?o2 ?to) (Different ?o2 ?o)
                 (PullConfig ?o ?p1 ?p2 ?to ?g)
            )
    :certified (CFreePullPose ?o ?p1 ?p2 ?to ?g ?o2 ?p22)
  )
  (:stream test-cfree-pull-traj-pose
    :inputs (?o ?to ?t ?o2 ?p2)
    :domain (and (Tool ?to) (Traj ?t) (Movable ?o) (Pose ?o2 ?p2) (Different ?o ?to) (Different ?o ?o2) (Different ?o2 ?to))
    :certified (CFreePullTrajPose ?o ?to ?t ?o2 ?p2)
  )
)
