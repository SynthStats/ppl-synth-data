module Submission
  ( State(..)
  , model
  , logLikRow
  ) where

import LazyPPL
import LazyPPL.Distributions
import TaskData (Row(..))

data State = State
  { b0 :: Double, b1 :: Double, b2 :: Double, noiseSigma :: Double
  } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  b0' <- sample $ normal 0 5
  b1' <- sample $ normal 0 5
  b2' <- sample $ normal 0 5
  s   <- sample $ gamma 2 1
  mapM_ (\r ->
    let predicted = b0' + b1' * x1Val r + b2' * x2Val r
    in score $ normalPdf predicted s (yVal r)) rows
  return $ State b0' b1' b2' s

logLikRow :: State -> Row -> Double
logLikRow (State b0' b1' b2' s) r =
  let predicted = b0' + b1' * x1Val r + b2' * x2Val r
  in normalLogPdf predicted s (yVal r)
