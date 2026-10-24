module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

data State = State Double Double Double Double Double Double
  deriving (Show)

model :: [Row] -> Meas State
model rows = do
  b0 <- sample (normal 0 2.0)
  b1 <- sample (normal 0 1.0)
  b2 <- sample (normal 0 1.0)
  b3 <- sample (normal 0 0.5)
  b4 <- sample (normal 0 0.5)
  b5 <- sample (normal 0 0.5)
  let st = State b0 b1 b2 b3 b4 b5

  let cell = 25.0
      xs = [cell / 2.0, 3.0 * cell / 2.0 .. 500.0 - cell / 2.0]
      ys = [cell / 2.0, 3.0 * cell / 2.0 .. 500.0 - cell / 2.0]
      areaElement = cell * cell
      integ = sum [intensityAt st xg yg * areaElement | xg <- xs, yg <- ys]

  score (exp (negate integ))
  mapM_ (\r -> score (exp (logLikRow st r))) rows
  return st

logLikRow :: State -> Row -> Double
logLikRow st row =
  let lam = max 1e-12 (intensityAt st (x row) (y row))
   in log lam

scaleCoord :: Double -> Double
scaleCoord z = (z - 250.0) / 250.0

intensityAt :: State -> Double -> Double -> Double
intensityAt (State b0 b1 b2 b3 b4 b5) x0 y0 =
  let x' = scaleCoord x0
      y' = scaleCoord y0
      eta = b0 + b1 * x' + b2 * y' + b3 * x' * x' + b4 * y' * y' + b5 * x' * y'
   in exp eta