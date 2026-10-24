module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

data State = State
  { intercept :: Double
  , weights :: [Double]
  , lengthScale :: Double
  , noiseSigma :: Double
  }
  deriving (Show)

model :: [Row] -> Meas State
model rows = do
  b0 <- sample (normal 0.0 100.0)
  w1 <- sample (normal 0.0 50.0)
  w2 <- sample (normal 0.0 50.0)
  w3 <- sample (normal 0.0 50.0)
  w4 <- sample (normal 0.0 50.0)
  w5 <- sample (normal 0.0 50.0)
  w6 <- sample (normal 0.0 50.0)
  ellRaw <- sample (gamma 2.0 5.0)
  sigmaRaw <- sample (gamma 2.0 5.0)

  let ell = max 1e-3 ellRaw
      sigma = max 1e-3 sigmaRaw
      state =
        State
          { intercept = b0
          , weights = [w1, w2, w3, w4, w5, w6]
          , lengthScale = ell
          , noiseSigma = sigma
          }
      rowScore r = exp (logLikRow state r)

  mapM_ (\r -> score (max 1e-300 (rowScore r))) rows
  return state

logLikRow :: State -> Row -> Double
logLikRow state row =
  let x = position row
      y = intensity row
      ell = max 1e-3 (lengthScale state)
      sigma = max 1e-3 (noiseSigma state)
      cs = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
      ws = weights state
      basis c =
        let z = (x - c) / ell
         in exp (-0.5 * z * z)
      mu =
        intercept state
          + sum (zipWith (\w c -> w * basis c) ws cs)
      ll = normalLogPdf mu sigma y
   in if isNaN ll || isInfinite ll then log 1e-300 else ll