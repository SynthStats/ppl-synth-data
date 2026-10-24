module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

type State = [Int]

model :: [Row] -> Meas State
model rows = do
  c0 <- sample (categorical [1,1,1,1])
  c1 <- sample (categorical [1,1,1,1])
  c2 <- sample (categorical [1,1,1,1])
  c3 <- sample (categorical [1,1,1,1])
  c4 <- sample (categorical [1,1,1,1])
  c5 <- sample (categorical [1,1,1,1])
  c6 <- sample (categorical [1,1,1,1])
  c7 <- sample (categorical [1,1,1,1])

  let st = [c0, c1, c2, c3, c4, c5, c6, c7]
      eps = 1e-9
      neq a b = if a /= b then 1.0 else eps
      edgeLik =
           neq c0 c1
        *  neq c0 c2
        *  neq c1 c3
        *  neq c2 c3
        *  neq c3 c4
        *  neq c4 c5
        *  neq c5 c6
        *  neq c6 c7
        *  neq c4 c7

  score edgeLik
  mapM_ (\r -> score (exp (logLikRow st r))) rows
  return st

logLikRow :: State -> Row -> Double
logLikRow state row =
  let eps = 1e-9
      getAt xs i =
        if i < 0 || i >= length xs then Nothing else Just (xs !! i)
      nums = map read (words (show row)) :: [Int]
  in case nums of
       [region, color] ->
         case getAt state region of
           Nothing -> log eps
           Just c -> if color >= 0 && color < 4 && c == color then 0.0 else log eps
       _ -> log eps