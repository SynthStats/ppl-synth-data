module Submission
  ( State(..)
  , model
  , logLikRow
  ) where

import LazyPPL
import LazyPPL.Distributions
import LazyPPL.Distributions.Memoization (memoize)
import TaskData (Row(..))

-- Two-state process: normal operation vs anomaly
data State = State
  { muNormal :: Double, sigmaNormal :: Double
  , muAnomaly :: Double, sigmaAnomaly :: Double
  , pAnomaly :: Double
  , stateFn :: Int -> Bool  -- time index -> is anomaly?
  } deriving (Show)

instance Show (Int -> Bool) where show _ = "<fn>"

model :: [Row] -> Meas State
model rows = do
  mN <- sample $ normal 100 10
  sN <- sample $ gamma 3 5
  mA <- sample $ normal 150 20
  sA <- sample $ gamma 3 10
  pA <- sample $ beta 1 9  -- ~10% anomaly rate
  anomalyFn <- sample $ memoize $ const $ bernoulli pA
  mapM_ (\r -> do
    let isA = anomalyFn (timeIdx r)
        (m, s) = if isA then (mA, sA) else (mN, sN)
    score $ normalPdf m s (measurement r)
    ) rows
  return $ State mN sN mA sA pA anomalyFn

logLikRow :: State -> Row -> Double
logLikRow st r =
  let mN = muNormal st; sN = sigmaNormal st
      mA = muAnomaly st; sA = sigmaAnomaly st
      pA = pAnomaly st
      likN = normalPdf mN sN (measurement r)
      likA = normalPdf mA sA (measurement r)
  in log (pA * likA + (1 - pA) * likN)
