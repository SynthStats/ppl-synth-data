module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

data State = State
  { stateB0 :: Double
  , stateB1 :: Double
  , stateB2 :: Double
  , stateB3 :: Double
  , stateRelNoise :: Double
  } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  b0 <- sample (normal 0 10)
  b1 <- sample (normal 0 10)
  b2 <- sample (normal 0 10)
  b3 <- sample (normal 0 10)
  relNoise <- sample (gamma 2 0.05)
  let st = State b0 b1 b2 b3 relNoise
  mapM_ (\r -> score (exp (logLikRow st r))) rows
  return st

logLikRow :: State -> Row -> Double
logLikRow st row =
  let xRaw = angle row
      y = torque row
      x = (xRaw - 90.0) / 90.0
      mu =
        let b0 = stateB0 st
            b1 = stateB1 st
            b2 = stateB2 st
            b3 = stateB3 st
        in b0 + b1 * x + b2 * x * x + b3 * x * x * x
      rn = max 1e-6 (stateRelNoise st)
      sigma = max 1e-6 (rn * max 1e-3 (abs mu))
  in normalLogPdf mu sigma y