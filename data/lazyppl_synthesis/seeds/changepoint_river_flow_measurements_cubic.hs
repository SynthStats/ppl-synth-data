module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

data State = State
  { lowMeanS  :: Double
  , highMeanS :: Double
  , midpointS :: Double
  , widthS    :: Double
  , sigmaS    :: Double
  } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  lowMean  <- sample (normal 60.0 50.0)
  highMean <- sample (normal 180.0 80.0)
  midpoint <- sample (uniformbounded 1.0 180.0)
  width    <- sample (gamma 2.0 7.0)
  sigma    <- sample (gamma 2.0 15.0)

  let st = State
        { lowMeanS = lowMean
        , highMeanS = highMean
        , midpointS = midpoint
        , widthS = width
        , sigmaS = sigma
        }

  mapM_ (\r -> score (exp (logLikRow st r))) rows
  return st

logLikRow :: State -> Row -> Double
logLikRow st row =
  let t = day row
      y = flow row
      lowM  = lowMeanS st
      highM = highMeanS st
      mid   = midpointS st
      wRaw  = widthS st
      sRaw  = sigmaS st
      w = max 1e-6 wRaw
      s = max 1e-6 sRaw
      z = (t - mid) / w
      trans = 1.0 / (1.0 + exp (-z))
      mu = lowM + (highM - lowM) * trans
  in normalLogPdf mu s y