(define (domain pick-and-place)
  (:requirements :strips :equality)
  (:predicates
    (Stackable ?o ?r)
    (Sink ?r)
    (Stove ?r)
    (Movable ?o)
    (PullTool ?o)
    (PushTool ?o)
    (Manipulanda ?o)
    (Graspable ?o)
    (Different ?o1 ?o2)
    (Floor ?r)

    (Pull ?o)
    (Push ?o)

    (Pose ?o ?p)
    (PullToolGoalPose ?o ?p)
    (PushToolGoalPose ?o ?p)
    (Grasp ?o ?g)
    (Conf ?q)
    (Kin ?o ?p ?g ?q ?t)
    (PullConfigKin ?o ?p1 ?p2 ?p3 ?to ?g ?top ?q1 ?q2 ?t)
    (PullConfig ?o ?p1 ?p2 ?to ?g)
    (PushConfigKin ?o ?p1 ?p2 ?p3 ?to ?g ?top ?q1 ?q2 ?t)
    (PushConfig ?o ?p1 ?p2 ?to ?g)
    (FreeMotion ?q1 ?t ?q2)
    (HoldingMotion ?q1 ?t ?q2 ?o ?g)
    (Supported ?o ?p ?r)
    (Traj ?t)

    (TrajCollision ?t ?o2 ?p2)
    (CFreePosePose ?o ?p ?o2 ?p2)
    (CFreeApproachPose ?o ?p ?g ?o2 ?p2)
    (CFreeTrajPose ?t ?o2 ?p2)
    (CFreePullPose ?o ?p1 ?p2 ?to ?g ?o2 ?p22)
    (CFreePullTrajPose ?o ?to ?t ?o2 ?p2)
    (CFreePushPose ?o ?p1 ?p2 ?to ?g ?o2 ?p22)
    (CFreePushTrajPose ?o ?to ?t ?o2 ?p2)

    (AtPose ?o ?p)
    (AtGrasp ?o ?g)
    (HandEmpty)
    (AtConf ?q)
    (CanMove)
    (Cleaned ?o)
    (Cooked ?o)

    (On ?o ?r)
    (Holding ?o)

    (UnsafePose ?o ?p)
    (UnsafeApproach ?o ?p ?g)
    (UnsafePull ?o ?p1 ?p2 ?to ?g)
    (UnsafePush ?o ?p1 ?p2 ?to ?g)
    (UnsafeTraj ?t)
    (UnsafePullTraj ?o ?to ?t)
    (UnsafePushTraj ?o ?to ?t)

  )

  (:action move_free
    :parameters (?q1 ?q2 ?t)
    :precondition (and (FreeMotion ?q1 ?t ?q2)
                       (AtConf ?q1) (HandEmpty) (CanMove)
                       ;(not (UnsafeTraj ?t))
                  )
    :effect (and (AtConf ?q2)
                 (not (AtConf ?q1)) (not (CanMove)))
  )
  (:action move_holding
    :parameters (?q1 ?q2 ?o ?g ?t)
    :precondition (and (HoldingMotion ?q1 ?t ?q2 ?o ?g)
                       (AtConf ?q1) (AtGrasp ?o ?g) (CanMove)
                       ;(not (UnsafeTraj ?t))
                  )
    :effect (and (AtConf ?q2)
                 (not (AtConf ?q1)) (not (CanMove)))
  )

  (:action pick
    :parameters (?o ?p ?g ?q ?t)
    :precondition (and (Kin ?o ?p ?g ?q ?t)
                       (AtPose ?o ?p) (HandEmpty) (AtConf ?q) (Movable ?o)
                       (not (UnsafeApproach ?o ?p ?g))
                       (not (UnsafeTraj ?t))
                  )
    :effect (and (AtGrasp ?o ?g) (CanMove)
                 (not (AtPose ?o ?p)) (not (HandEmpty)))
  )

  (:action place
    :parameters (?o ?p ?g ?q ?t)
    :precondition (and (Kin ?o ?p ?g ?q ?t)
                       (AtGrasp ?o ?g) (AtConf ?q)
                       (not (UnsafePose ?o ?p))
                       (not (UnsafeApproach ?o ?p ?g))
                       (not (UnsafeTraj ?t))
                  )
    :effect (and (AtPose ?o ?p) (HandEmpty) (CanMove)
                 (not (AtGrasp ?o ?g)))
  )

  (:action pull
    :parameters (?o ?p1 ?p2 ?p3 ?to ?g ?q1 ?q2 ?top ?t)
    :precondition (and (AtPose ?o ?p1)
                       (Manipulanda ?o)
                       (PullToolGoalPose ?o ?p2)
                       (PullConfigKin ?o ?p1 ?p2 ?p3 ?to ?g ?top ?q1 ?q2 ?t)
                       (AtGrasp ?to ?g)
                       (PullTool ?to)
                       (AtConf ?q1)
                       (not (= ?o ?to))
                       (not (UnsafePose ?o ?p2))
                       (not (UnsafePose ?to ?top2))
                       (not (UnsafePose ?to ?top))
                       (not (UnsafePull ?o ?p1 ?p3 ?to ?g))
                       (not (UnsafePullTraj ?o ?to ?t))
                  )
    :effect (and (AtPose ?o ?p3) (not (AtPose ?o ?p1))
                 (AtPose ?to ?top)
                 (not (AtGrasp ?to ?g)) (HandEmpty) (CanMove)
                 (not (AtConf ?q1)) (AtConf ?q2)
                 (Pull ?to)
            )
  )

  (:action push
    :parameters (?o ?p1 ?p2 ?p3 ?to ?g ?q1 ?q2 ?top ?t)
    :precondition (and (AtPose ?o ?p1)
                       (Manipulanda ?o)
                       (PushToolGoalPose ?o ?p2)
                       (PushConfigKin ?o ?p1 ?p2 ?p3 ?to ?g ?top ?q1 ?q2 ?t)
                       (AtGrasp ?to ?g)
                       (PushTool ?to)
                       (AtConf ?q1)
                       (not (= ?o ?to))
                       (not (UnsafePose ?o ?p2))
                       (not (UnsafePose ?to ?top2))
                       (not (UnsafePose ?to ?top))
                       (not (UnsafePush ?o ?p1 ?p3 ?to ?g))
                       (not (UnsafePushTraj ?o ?to ?t))
                  )
    :effect (and (AtPose ?o ?p3) (not (AtPose ?o ?p1))
                 (AtPose ?to ?top)
                 (not (AtGrasp ?to ?g)) (HandEmpty) (CanMove)
                 (not (AtConf ?q1)) (AtConf ?q2)
                 (Push ?to)
            )
  )

  (:action clean
    :parameters (?o ?r)
    :precondition (and (Stackable ?o ?r) (Sink ?r)
                       (On ?o ?r))
    :effect (Cleaned ?o)
  )
  (:action cook
    :parameters (?o ?r)
    :precondition (and (Stackable ?o ?r) (Stove ?r)
                       (On ?o ?r) (Cleaned ?o))
    :effect (and (Cooked ?o)
                 (not (Cleaned ?o)))
  )

  (:derived (On ?o ?r)
    (exists (?p) (and (Supported ?o ?p ?r)
                      (AtPose ?o ?p)))
  )
  (:derived (Holding ?o)
    (exists (?g) (and (Grasp ?o ?g)
                      (AtGrasp ?o ?g)))
  )

  (:derived (UnsafePose ?o ?p)
    (exists (?o2 ?p2) (and (Pose ?o ?p) (Pose ?o2 ?p2) (not (= ?o ?o2))
                           (not (CFreePosePose ?o ?p ?o2 ?p2))
                           (AtPose ?o2 ?p2)))
  )
  (:derived (UnsafeApproach ?o ?p ?g)
    (exists (?o2 ?p2) (and (Pose ?o ?p) (Grasp ?o ?g) (Pose ?o2 ?p2) (not (= ?o ?o2))
                           (not (CFreeApproachPose ?o ?p ?g ?o2 ?p2))
                           (AtPose ?o2 ?p2)))
  )
  (:derived (UnsafePull ?o ?p1 ?p2 ?to ?g)
    (exists (?o2 ?p22) (and (Pose ?o ?p1) (Pose ?o ?p2)
                            (PullConfig ?o ?p1 ?p2 ?to ?g)
                            (PullTool ?to)
                            (Pose ?o2 ?p22) (not (= ?o ?o2)) (not (= ?o ?to)) (not (= ?o2 ?to))
                            (not (CFreePullPose ?o ?p1 ?p2 ?to ?g ?o2 ?p22))
                            (AtPose ?o2 ?p22)
                       )
    )
  )
  (:derived (UnsafePush ?o ?p1 ?p2 ?to ?g)
    (exists (?o2 ?p22) (and (Pose ?o ?p1) (Pose ?o ?p2)
                            (PushConfig ?o ?p1 ?p2 ?to ?g)
                            (PushTool ?to)
                            (Pose ?o2 ?p22) (not (= ?o ?o2)) (not (= ?o ?to)) (not (= ?o2 ?to))
                            (not (CFreePushPose ?o ?p1 ?p2 ?to ?g ?o2 ?p22))
                            (AtPose ?o2 ?p22)
                       )
    )
  )
  (:derived (UnsafeTraj ?t)
    (exists (?o2 ?p2) (and (Traj ?t) (Pose ?o2 ?p2)
                           (not (CFreeTrajPose ?t ?o2 ?p2))
                           ; (TrajCollision ?t ?o2 ?p2)
                           (AtPose ?o2 ?p2)))
  )
  (:derived (UnsafePullTraj ?o ?to ?t)
    (exists (?o2 ?p2) (and (Traj ?t) (Pose ?o2 ?p2)
                           (PullTool ?to) (Movable ?o)
                           (not (= ?o ?o2)) (not (= ?to ?o2)) (not (= ?o ?to))
                           (not (CFreePullTrajPose ?o ?to ?t ?o2 ?p2))
                           (AtPose ?o2 ?p2)
                      )
    )    
  )
  (:derived (UnsafePushTraj ?o ?to ?t)
    (exists (?o2 ?p2) (and (Traj ?t) (Pose ?o2 ?p2)
                           (PushTool ?to) (Movable ?o)
                           (not (= ?o ?o2)) (not (= ?to ?o2)) (not (= ?o ?to))
                           (not (CFreePushTrajPose ?o ?to ?t ?o2 ?p2))
                           (AtPose ?o2 ?p2)
                      )
    )    
  )

)
