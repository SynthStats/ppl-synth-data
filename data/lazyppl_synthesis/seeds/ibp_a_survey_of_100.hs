module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

data State = State
  { stateBias :: Double
  , stateNoise :: Double
  , stateWeights :: [Double]
  } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  bias <- sample (normal 0 3)
  noise <- sample (gamma 2 1)
  w1 <- sample (normal 0 2)
  w2 <- sample (normal 0 2)
  w3 <- sample (normal 0 2)
  w4 <- sample (normal 0 2)
  w5 <- sample (normal 0 2)
  w6 <- sample (normal 0 2)
  let st = State bias noise [w1, w2, w3, w4, w5, w6]
  mapM_ (\r -> score (exp (logLikRow st r))) rows
  return st

logLikRow :: State -> Row -> Double
logLikRow st row =
  let eps = 1e-6
      sig = max eps (stateNoise st)
      ws = stateWeights st
      x = rowFlags row
      mu = stateBias st + x * sum ws
  in normalLogPdf mu sig (rowRating row)