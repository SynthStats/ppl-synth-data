module Submission
  ( State(..)
  , model
  , logLikRow
  ) where

import LazyPPL
import LazyPPL.Distributions
import TaskData (Row(..))

-- Bimodal restaurant ratings: satisfied (high) vs dissatisfied (low)
data State = State
  { muHigh :: Double, muLow :: Double
  , sigmaHigh :: Double, sigmaLow :: Double
  , pSatisfied :: Double
  } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  mH <- sample $ normal 4.0 0.5
  mL <- sample $ normal 2.0 0.5
  sH <- sample $ gamma 3 0.3
  sL <- sample $ gamma 3 0.3
  p  <- sample $ beta 3 2  -- ~60% satisfied
  mapM_ (\r -> do
    satisfied <- sample $ bernoulli p
    let (m, s) = if satisfied then (mH, sH) else (mL, sL)
    score $ normalPdf m s (rating r)
    ) rows
  return $ State mH mL sH sL p

logLikRow :: State -> Row -> Double
logLikRow (State mH mL sH sL p) r =
  log (p * normalPdf mH sH (rating r) + (1 - p) * normalPdf mL sL (rating r))
