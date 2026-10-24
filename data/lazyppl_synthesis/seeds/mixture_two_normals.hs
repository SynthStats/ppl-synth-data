module Submission
  ( State(..)
  , model
  , logLikRow
  ) where

import LazyPPL
import LazyPPL.Distributions
import TaskData (Row(..))

data State = State
  { mu1 :: Double, mu2 :: Double
  , sigma1 :: Double, sigma2 :: Double
  , mixWeight :: Double
  } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  m1 <- sample $ normal 0 10
  m2 <- sample $ normal 0 10
  s1 <- sample $ gamma 2 1
  s2 <- sample $ gamma 2 1
  w  <- sample $ beta 2 2
  mapM_ (\r ->
    let x = value r
        lik = w * normalPdf m1 s1 x + (1 - w) * normalPdf m2 s2 x
    in score lik) rows
  return $ State m1 m2 s1 s2 w

logLikRow :: State -> Row -> Double
logLikRow (State m1 m2 s1 s2 w) r =
  let x = value r
  in log (w * normalPdf m1 s1 x + (1 - w) * normalPdf m2 s2 x)
