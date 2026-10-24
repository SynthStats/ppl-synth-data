module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

data State = State
  { muRest :: Double
  , muStress :: Double
  , obsSigma :: Double
  , stayRest :: Double
  , stayStress :: Double
  , initRest :: Double
  } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  muRest0 <- sample (normal 70.0 15.0)
  gap <- sample (gamma 2.0 15.0)
  sigma0 <- sample (gamma 2.0 3.0)
  pStayRest <- sample (beta 8.0 2.0)
  pStayStress <- sample (beta 8.0 2.0)
  piRest0 <- sample (beta 1.0 1.0)
  let muStress0 = muRest0 + gap
      sigma = max 1e-3 sigma0
      state =
        State
          { muRest = muRest0
          , muStress = muStress0
          , obsSigma = sigma
          , stayRest = pStayRest
          , stayStress = pStayStress
          , initRest = piRest0
          }

      logSumExp :: Double -> Double -> Double
      logSumExp a b =
        let m = max a b
        in if isInfinite m && m < 0
             then m
             else m + log (exp (a - m) + exp (b - m))

      step :: (Double, Double) -> Row -> (Double, Double)
      step (lrPrev, lsPrev) row =
        let y = heartRate row
            eRest = normalLogPdf muRest0 sigma y
            eStress = normalLogPdf muStress0 sigma y
            lr =
              eRest
                + logSumExp
                    (lrPrev + log (max 1e-12 pStayRest))
                    (lsPrev + log (max 1e-12 (1.0 - pStayStress)))
            ls =
              eStress
                + logSumExp
                    (lrPrev + log (max 1e-12 (1.0 - pStayRest)))
                    (lsPrev + log (max 1e-12 pStayStress))
        in (lr, ls)

      totalLogLik :: Double
      totalLogLik =
        case rows of
          [] -> 0.0
          (r:rs) ->
            let y0 = heartRate r
                lr0 = log (max 1e-12 piRest0) + normalLogPdf muRest0 sigma y0
                ls0 = log (max 1e-12 (1.0 - piRest0)) + normalLogPdf muStress0 sigma y0
                (lrF, lsF) = foldl step (lr0, ls0) rs
            in logSumExp lrF lsF

  score (exp totalLogLik)
  return state

logLikRow :: State -> Row -> Double
logLikRow state row =
  let mr = muRest state
      ms = muStress state
      s = max 1e-3 (obsSigma state)
      p0 = min 0.999999 (max 1e-6 (initRest state))
      y = heartRate row
      a = log p0 + normalLogPdf mr s y
      b = log (1.0 - p0) + normalLogPdf ms s y
      m = max a b
  in if isInfinite m && m < 0
       then -1e300
       else m + log (exp (a - m) + exp (b - m))