module Submission
  ( State(..)
  , model
  , logLikRow
  ) where

import LazyPPL
import LazyPPL.Distributions
import TaskData (Row(..))

data State = State
  { b0 :: Double, b1 :: Double
  } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  b0' <- sample $ normal 0 2
  b1' <- sample $ normal 0 1
  mapM_ (\r ->
    let logRate = b0' + b1' * xVal r
        rate = exp logRate
    in score $ poissonPdf rate (count r)) rows
  return $ State b0' b1'

logLikRow :: State -> Row -> Double
logLikRow (State b0' b1') r =
  let rate = exp (b0' + b1' * xVal r)
  in poissonLogPmf rate (count r)
