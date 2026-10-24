module Submission
  ( State(..)
  , model
  , logLikRow
  ) where

import LazyPPL
import LazyPPL.Distributions
import TaskData (Row(..))

data State = State { p :: Double } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  prob <- sample $ beta 2 5
  mapM_ (\r ->
    score $ exp (binomialLogPmf (nTrials r) prob (nSuccesses r))) rows
  return $ State prob

logLikRow :: State -> Row -> Double
logLikRow (State prob) r = binomialLogPmf (nTrials r) prob (nSuccesses r)
