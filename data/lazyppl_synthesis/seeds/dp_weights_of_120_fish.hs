module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

data State = State [Double] [Double] [Double]
  deriving (Show)

model :: [Row] -> Meas State
model rows = do
  let k = 8
      dirAlpha = replicate k 0.5
  mixWeights <- sample (dirichlet dirAlpha)
  mus <- mapM (\_ -> sample (normal 0.0 50.0)) [1 .. k]
  sigmasRaw <- mapM (\_ -> sample (gamma 2.0 5.0)) [1 .. k]
  let sigmas = map (\s -> if s <= 1e-6 then 1e-6 else s) sigmasRaw
      st = State mixWeights mus sigmas
  mapM_ (\r -> score (exp (logLikRow st r))) rows
  return st

logLikRow :: State -> Row -> Double
logLikRow (State mixWeights mus sigmas) row =
  let x = weight row
      safeSigma s = if s <= 1e-6 then 1e-6 else s
      safeW w = if w <= 1e-300 then 1e-300 else w
      lterms =
        zipWith3
          (\w mu sg -> log (safeW w) + normalLogPdf mu (safeSigma sg) x)
          mixWeights
          mus
          sigmas
      ll =
        if null lterms
          then log 1e-300
          else
            let m = maximum lterms
                s = sum (map (\t -> exp (t - m)) lterms)
             in m + log (if s <= 1e-300 then 1e-300 else s)
   in if isNaN ll || isInfinite ll then log 1e-300 else ll