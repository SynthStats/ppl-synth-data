module Submission where

import LazyPPL
import LazyPPL.Distributions hiding (beta)
import TaskData

data State = State
  { mu0 :: Double
  , delta :: Double
  , alpha :: Double
  , beta :: Double
  , eventTimes :: [Double]
  } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  mu0'   <- sample (gamma 2.0 1.0)
  delta' <- sample (gamma 2.0 1.0)
  alpha' <- sample (gamma 2.0 1.0)
  beta'  <- sample (gamma 2.0 1.0)
  let ts = map time rows
      st = State
        { mu0 = mu0'
        , delta = delta'
        , alpha = alpha'
        , beta = beta'
        , eventTimes = ts
        }
      tMax = 24.0
      eps = 1e-12
      d = max eps (delta st)
      b = max eps (beta st)
      eventTerm = sum (map (logLikRow st) rows)
      baseComp = mu0 st / d * (1.0 - exp (-(d * tMax)))
      exciteComp = sum
        [ alpha st / b * (1.0 - exp (-(b * max 0.0 (tMax - ti))))
        | ti <- ts
        ]
      ll = eventTerm - baseComp - exciteComp
  score (max 1e-300 (exp ll))
  return st

logLikRow :: State -> Row -> Double
logLikRow st row =
  let t = time row
      eps = 1e-12
      d = max eps (delta st)
      b = max eps (beta st)
      base = mu0 st * exp (-(d * max 0.0 t))
      excite = sum
        [ alpha st * exp (-(b * (t - ti)))
        | ti <- eventTimes st
        , ti < t
        ]
      lam = max eps (base + excite)
  in log lam