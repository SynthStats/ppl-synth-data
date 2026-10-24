module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

data State = State
  { intercept :: Double
  , seasonSin :: Double
  , seasonCos :: Double
  }
  deriving (Show)

model :: [Row] -> Meas State
model rows = do
  intercept' <- sample (normal 0 2)
  seasonSin' <- sample (normal 0 1)
  seasonCos' <- sample (normal 0 1)
  let state =
        State
          { intercept = intercept'
          , seasonSin = seasonSin'
          , seasonCos = seasonCos'
          }
  mapM_
    (\row ->
        let ll = logLikRow state row
            w = if isNaN ll || isInfinite ll || ll < (-1.0e300) then 1.0e-300 else exp ll
         in score w)
    rows
  return state

logLikRow :: State -> Row -> Double
logLikRow state row =
  let tRaw = time row
      t = if isNaN tRaw || isInfinite tRaw then 0 else tRaw
      eta =
        intercept state
          + seasonSin state * sin (2 * pi * t)
          + seasonCos state * cos (2 * pi * t)
      rate =
        let r = exp eta
         in if isNaN r || isInfinite r || r <= 0 then 1.0e-12 else r
      y = 1 :: Int
      ll = poissonLogPmf rate y
   in if isNaN ll || isInfinite ll then log 1.0e-300 else ll