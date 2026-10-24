module Submission
  ( State(..)
  , model
  , logLikRow
  ) where

import LazyPPL
import LazyPPL.Distributions
import TaskData (Row(..))

-- Von Bertalanffy growth: L(t) = alpha - beta * lambda^t
data State = State
  { alpha :: Double, betaParam :: Double
  , lambdaParam :: Double, sigma :: Double
  } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  a   <- sample $ normal 3 1
  b   <- sample $ normal 1 1
  lam <- sample $ beta 2 2
  s   <- sample $ gamma 2 0.5
  mapM_ (\r ->
    let predicted = a - b * (lam ** age r)
    in score $ normalPdf predicted s (yLength r)) rows
  return $ State a b lam s

logLikRow :: State -> Row -> Double
logLikRow (State a b lam s) r =
  let predicted = a - b * (lam ** age r)
  in normalLogPdf predicted s (yLength r)
