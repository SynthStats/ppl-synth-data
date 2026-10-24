module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

data State = State
  { c0 :: Double
  , k :: Double
  , sigma :: Double
  } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  c0' <- sample (gamma 2.0 2.0)
  k' <- sample (gamma 2.0 1.0)
  sigma0 <- sample (gamma 2.0 1.0)
  let sigma' = max 1e-6 sigma0
      state = State { c0 = c0', k = k', sigma = sigma' }
  mapM_ (\row -> score (exp (logLikRow state row))) rows
  return state

logLikRow :: State -> Row -> Double
logLikRow state row =
  let x = getX row
      y = getY row
      c0' = max 1e-12 (c0 state)
      k' = max 1e-12 (k state)
      sigma' = max 1e-6 (sigma state)
      mu = c0' * exp (-(k' * x))
      ll = normalLogPdf mu sigma' y
  in if isNaN ll || isInfinite ll then log 1e-300 else ll