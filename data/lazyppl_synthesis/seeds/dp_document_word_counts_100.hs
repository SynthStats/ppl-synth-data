module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

type State = ([Double], [Double])

model :: [Row] -> Meas State
model rows = do
  let k = 8
      alpha = replicate k 0.5
  weights <- sample (dirichlet alpha)
  rates <- sequence [sample (gamma 2.0 50.0) | _ <- [1 .. k]]
  let state = (weights, rates)
  mapM_ (\row -> score (exp (logLikRow state row))) rows
  return state

logLikRow :: State -> Row -> Double
logLikRow (weights, rates) row =
  let x = max 0 (round (wordCount row))
      eps = 1e-300
      ws = map (\w -> max eps w) weights
      ls = map (\r -> max eps r) rates
      terms = zipWith (\w lam -> log w + poissonLogPmf lam x) ws ls
      m = if null terms then log eps else maximum terms
      s = sum (map (\t -> exp (t - m)) terms)
   in if null terms || s <= 0 || isNaN s || isInfinite s
        then log eps
        else m + log s