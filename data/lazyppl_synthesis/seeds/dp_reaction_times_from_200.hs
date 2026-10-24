module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

data State = State [Double] [Double] [Double]
  deriving (Show)

model :: [Row] -> Meas State
model rows = do
  let k = 5
      rts = map rt rows
      safeRT x = max 1e-6 x
      logRts = map (log . safeRT) rts
      n = max 1 (length logRts)
      center = sum logRts / fromIntegral n
      alpha = replicate k 0.5
  ws <- sample (dirichlet alpha)
  mus <- sequence [sample (normal center 2.0) | _ <- [1 .. k]]
  sigs0 <- sequence [sample (gamma 2.0 0.3) | _ <- [1 .. k]]
  let sigs = map (\s -> max 1e-3 s) sigs0
      st = State ws mus sigs
  mapM_ (\row -> score (exp (logLikRow st row))) rows
  return st

logLikRow :: State -> Row -> Double
logLikRow (State ws mus sigs) row =
  let x = rt row
      eps = 1e-12
      x' = max eps x
      safeSig s = max 1e-3 s
      comps = zip3 ws mus sigs
      logTerms =
        [ log (max eps w) + logNormalLogPdf mu (safeSig sg) x'
        | (w, mu, sg) <- comps
        ]
      m = if null logTerms then log eps else maximum logTerms
      s = sum [exp (t - m) | t <- logTerms]
   in if x <= 0 || null logTerms || s <= 0 || isNaN m || isInfinite m
        then log 1e-300
        else m + log s