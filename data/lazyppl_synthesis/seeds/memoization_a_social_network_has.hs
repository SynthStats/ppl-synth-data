module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

data State = State
  { alpha :: Double
  , influences :: [Double]
  }
  deriving (Show)

model :: [Row] -> Meas State
model rows = do
  a <- sample (normal 0 2)
  infls <- sequence [sample (normal 0 1) | _ <- [1 .. 40]]
  let st = State {alpha = a, influences = infls}
  mapM_ (\row -> score (exp (logLikRow st row))) rows
  return st

logLikRow :: State -> Row -> Double
logLikRow st row =
  let aIdx0 :: Int
      aIdx0 = round (userA row)

      aEff =
        if aIdx0 >= 0 && aIdx0 < length (influences st)
          then influences st !! aIdx0
          else 0.0

      eta = alpha st + aEff
      p0 = 1.0 / (1.0 + exp (-eta))
      eps = 1e-9
      p = max eps (min (1.0 - eps) p0)

      y = reposted row /= 0
   in bernoulliLogPmf p y