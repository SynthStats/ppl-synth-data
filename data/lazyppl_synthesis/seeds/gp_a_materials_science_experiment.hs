module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

data State = State
  { a0 :: Double
  , a1 :: Double
  , a2 :: Double
  , a3 :: Double
  , relNoise :: Double
  } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  a0' <- sample (normal 0 10)
  a1' <- sample (normal 0 10)
  a2' <- sample (normal 0 10)
  a3' <- sample (normal 0 10)
  relNoise' <- sample (gamma 2 0.02)
  let state =
        State
          { a0 = a0'
          , a1 = a1'
          , a2 = a2'
          , a3 = a3'
          , relNoise = relNoise'
          }
  mapM_ (\row -> score (exp (logLikRow state row))) rows
  return state

logLikRow :: State -> Row -> Double
logLikRow state row =
  let v = voltage row
      y = current row
      x = (v - 5.0) / 5.0
      mu =
        a0 state
          + a1 state * x
          + a2 state * x * x
          + a3 state * x * x * x
      eps = 1e-9
      rn = max eps (relNoise state)
      sigma = max eps (rn * max eps (abs mu))
      d = normalPdf mu sigma y
   in max 1e-300 d