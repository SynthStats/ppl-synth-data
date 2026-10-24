module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

data State = State
  { globalMean :: Double
  , betweenSd :: Double
  , withinSd :: Double
  , locationMeans :: [Double]
  } deriving (Show)

model :: [Row] -> Meas State
model rows = do
  mu0 <- sample (normal 0.0 20.0)
  tauRaw <- sample (gamma 2.0 5.0)
  sigmaRaw <- sample (gamma 2.0 5.0)
  let tau = max 1e-6 tauRaw
  let sigma = max 1e-6 sigmaRaw
  mu1  <- sample (normal mu0 tau)
  mu2  <- sample (normal mu0 tau)
  mu3  <- sample (normal mu0 tau)
  mu4  <- sample (normal mu0 tau)
  mu5  <- sample (normal mu0 tau)
  mu6  <- sample (normal mu0 tau)
  mu7  <- sample (normal mu0 tau)
  mu8  <- sample (normal mu0 tau)
  mu9  <- sample (normal mu0 tau)
  mu10 <- sample (normal mu0 tau)
  mu11 <- sample (normal mu0 tau)
  mu12 <- sample (normal mu0 tau)
  mu13 <- sample (normal mu0 tau)
  mu14 <- sample (normal mu0 tau)
  mu15 <- sample (normal mu0 tau)
  mu16 <- sample (normal mu0 tau)
  mu17 <- sample (normal mu0 tau)
  mu18 <- sample (normal mu0 tau)
  mu19 <- sample (normal mu0 tau)
  mu20 <- sample (normal mu0 tau)
  mu21 <- sample (normal mu0 tau)
  mu22 <- sample (normal mu0 tau)
  mu23 <- sample (normal mu0 tau)
  mu24 <- sample (normal mu0 tau)
  mu25 <- sample (normal mu0 tau)
  let state =
        State
          { globalMean = mu0
          , betweenSd = tau
          , withinSd = sigma
          , locationMeans =
              [ mu1, mu2, mu3, mu4, mu5
              , mu6, mu7, mu8, mu9, mu10
              , mu11, mu12, mu13, mu14, mu15
              , mu16, mu17, mu18, mu19, mu20
              , mu21, mu22, mu23, mu24, mu25
              ]
          }
  mapM_ (\r -> score (exp (logLikRow state r))) rows
  return state

logLikRow :: State -> Row -> Double
logLikRow state row =
  let sigma = max 1e-6 (withinSd state)
      mus = locationMeans state
      loc = round (locationId row) :: Int
      y = waitTime row
      idx0 = loc - 1
      valid = idx0 >= 0 && idx0 < length mus
   in if not valid
        then log 1e-300
        else max (log 1e-300) (normalLogPdf (mus !! idx0) sigma y)