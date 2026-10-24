module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

type State = (Double, Double, Double, [Double])
-- (global mean mu, between-driver sd tau, ride noise sd sigma, driver means)

model :: [Row] -> Meas State
model rows = do
  mu <- sample (normal 0.15 0.10)
  tauRaw <- sample (gamma 2.0 0.05)
  sigmaRaw <- sample (gamma 2.0 0.05)
  let tau = max 1e-6 tauRaw
      sigma = max 1e-6 sigmaRaw

  th0  <- sample (normal mu tau)
  th1  <- sample (normal mu tau)
  th2  <- sample (normal mu tau)
  th3  <- sample (normal mu tau)
  th4  <- sample (normal mu tau)
  th5  <- sample (normal mu tau)
  th6  <- sample (normal mu tau)
  th7  <- sample (normal mu tau)
  th8  <- sample (normal mu tau)
  th9  <- sample (normal mu tau)
  th10 <- sample (normal mu tau)
  th11 <- sample (normal mu tau)
  th12 <- sample (normal mu tau)
  th13 <- sample (normal mu tau)
  th14 <- sample (normal mu tau)
  th15 <- sample (normal mu tau)
  th16 <- sample (normal mu tau)
  th17 <- sample (normal mu tau)
  th18 <- sample (normal mu tau)
  th19 <- sample (normal mu tau)
  th20 <- sample (normal mu tau)
  th21 <- sample (normal mu tau)
  th22 <- sample (normal mu tau)
  th23 <- sample (normal mu tau)
  th24 <- sample (normal mu tau)
  th25 <- sample (normal mu tau)
  th26 <- sample (normal mu tau)
  th27 <- sample (normal mu tau)
  th28 <- sample (normal mu tau)
  th29 <- sample (normal mu tau)

  let thetas =
        [ th0, th1, th2, th3, th4, th5, th6, th7, th8, th9
        , th10, th11, th12, th13, th14, th15, th16, th17, th18, th19
        , th20, th21, th22, th23, th24, th25, th26, th27, th28, th29
        ]
      st = (mu, tau, sigma, thetas)

  mapM_ (\r -> score (exp (logLikRow st r))) rows
  return st

logLikRow :: State -> Row -> Double
logLikRow (mu, _tau, sigmaRaw, thetas) row =
  let sigma = max 1e-6 sigmaRaw
      didD = driverId row
      did = floor didD :: Int
      y = tipPct row
      theta =
        if did >= 0 && did < length thetas
          then thetas !! did
          else mu
      ySafe =
        if isNaN y || isInfinite y then 0.0 else y
      ll = normalLogPdf theta sigma ySafe
  in if isNaN ll || isInfinite ll then log 1e-300 else ll