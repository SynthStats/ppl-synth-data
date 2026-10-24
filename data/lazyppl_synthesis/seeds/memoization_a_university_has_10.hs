module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

data State = State
  { deptRates :: [Double]
  , rateShape :: Double
  , rateScale :: Double
  } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  shapeHyper <- sample (gamma 2.0 2.0)
  scaleHyper <- sample (gamma 2.0 2.0)
  lambda0 <- sample (gamma shapeHyper scaleHyper)
  lambda1 <- sample (gamma shapeHyper scaleHyper)
  lambda2 <- sample (gamma shapeHyper scaleHyper)
  lambda3 <- sample (gamma shapeHyper scaleHyper)
  lambda4 <- sample (gamma shapeHyper scaleHyper)
  lambda5 <- sample (gamma shapeHyper scaleHyper)
  lambda6 <- sample (gamma shapeHyper scaleHyper)
  lambda7 <- sample (gamma shapeHyper scaleHyper)
  lambda8 <- sample (gamma shapeHyper scaleHyper)
  lambda9 <- sample (gamma shapeHyper scaleHyper)

  let state =
        State
          { deptRates =
              [ lambda0
              , lambda1
              , lambda2
              , lambda3
              , lambda4
              , lambda5
              , lambda6
              , lambda7
              , lambda8
              , lambda9
              ]
          , rateShape = shapeHyper
          , rateScale = scaleHyper
          }

  let scoreRow r =
        let ll = logLikRow state r
            w = if ll < (-700.0) then 1e-300 else exp ll
         in score w

  mapM_ scoreRow rows
  return state

logLikRow :: State -> Row -> Double
logLikRow state row =
  let eps = 1e-9
      bad = log 1e-300
      rates = deptRates state
      d = round (department row) :: Int
      k = round (publicationCount row) :: Int
      lambda =
        if d >= 0 && d < length rates
          then max eps (rates !! d)
          else eps
   in if d < 0 || d >= length rates || k < 0
        then bad
        else poissonLogPmf lambda k