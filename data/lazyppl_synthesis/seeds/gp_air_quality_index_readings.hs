module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

data State = State
  { baseline :: Double
  , amplitude :: Double
  , rate :: Double
  , noiseSigma :: Double
  }
  deriving (Show)

model :: [Row] -> Meas State
model rows = do
  baseline' <- sample (normal 50.0 50.0)
  amplitude' <- sample (gamma 2.0 25.0)
  rate' <- sample (gamma 2.0 1.0)
  noiseSigma' <- sample (gamma 2.0 10.0)
  let state =
        State
          { baseline = baseline'
          , amplitude = amplitude'
          , rate = rate'
          , noiseSigma = noiseSigma'
          }
  mapM_
    (\row ->
       let ll = logLikRow state row
           w = exp ll
        in score (if w > 0.0 && not (isNaN w) && not (isInfinite w) then w else 1e-300))
    rows
  return state

logLikRow :: State -> Row -> Double
logLikRow state row =
  let d = max 0.0 (distance row)
      y = aqi row
      b = baseline state
      a = max 1e-9 (amplitude state)
      r = max 1e-9 (rate state)
      s = max 1e-9 (noiseSigma state)
      mu = b + a * exp (-(r * d))
      ll = normalLogPdf mu s y
   in if isNaN ll || isInfinite ll then log 1e-300 else ll