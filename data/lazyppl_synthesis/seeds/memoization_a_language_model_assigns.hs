module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

data State = State
  { embeddings :: [Double]
  , noiseSigma :: Double
  } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  obsSigmaRaw <- sample (gamma 2.0 1.0)
  let obsSigma = max 1e-6 obsSigmaRaw
  e0  <- sample (normal 0.0 3.0)
  e1  <- sample (normal 0.0 3.0)
  e2  <- sample (normal 0.0 3.0)
  e3  <- sample (normal 0.0 3.0)
  e4  <- sample (normal 0.0 3.0)
  e5  <- sample (normal 0.0 3.0)
  e6  <- sample (normal 0.0 3.0)
  e7  <- sample (normal 0.0 3.0)
  e8  <- sample (normal 0.0 3.0)
  e9  <- sample (normal 0.0 3.0)
  e10 <- sample (normal 0.0 3.0)
  e11 <- sample (normal 0.0 3.0)
  e12 <- sample (normal 0.0 3.0)
  e13 <- sample (normal 0.0 3.0)
  e14 <- sample (normal 0.0 3.0)
  e15 <- sample (normal 0.0 3.0)
  e16 <- sample (normal 0.0 3.0)
  e17 <- sample (normal 0.0 3.0)
  e18 <- sample (normal 0.0 3.0)
  e19 <- sample (normal 0.0 3.0)
  e20 <- sample (normal 0.0 3.0)
  e21 <- sample (normal 0.0 3.0)
  e22 <- sample (normal 0.0 3.0)
  e23 <- sample (normal 0.0 3.0)
  e24 <- sample (normal 0.0 3.0)

  let st = State
        { embeddings =
            [ e0, e1, e2, e3, e4
            , e5, e6, e7, e8, e9
            , e10, e11, e12, e13, e14
            , e15, e16, e17, e18, e19
            , e20, e21, e22, e23, e24
            ]
        , noiseSigma = obsSigma
        }

  let loop [] = return ()
      loop (r:rs) = do
        score (exp (logLikRow st r))
        loop rs

  loop rows
  return st

rowWordIndex :: Row -> Int
rowWordIndex row = case show row of
  s -> case reads s of
    [(k, _)] -> k
    _ -> 0

rowMeasurement :: Row -> Double
rowMeasurement row = case reverse (words (map sep (show row))) of
  (tok:_) -> case reads tok of
    [(y, _)] -> y
    _ -> 0.0
  _ -> 0.0
  where
    sep c = if c == '(' || c == ')' || c == ',' then ' ' else c

logLikRow :: State -> Row -> Double
logLikRow st row =
  let sigma = max 1e-6 (noiseSigma st)
      ws = embeddings st
      k = rowWordIndex row
      mu =
        if k >= 0 && k < length ws
          then ws !! k
          else 0.0
      y = rowMeasurement row
      ll = normalLogPdf mu sigma y
  in if isNaN ll || isInfinite ll then log 1e-300 else ll