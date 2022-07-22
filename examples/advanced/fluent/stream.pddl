(define (stream fluent)
  (:stream sample-pickable
    :inputs (?b)
    :domain (Block ?b)
    :fluents (OnTable)
    :outputs (?t)
    :certified (Pickable ?b ?t)
  )
  (:stream test-cleanable
    :inputs (?b)
    :domain (Block ?b)
    :fluents (OnTable)
    :certified (Cleanable ?b)
  )
)