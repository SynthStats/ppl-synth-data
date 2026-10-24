module Submission where

import LazyPPL
import LazyPPL.Distributions
import TaskData

data State = State
  { globalMean :: Double
  , countySd :: Double
  , obsSd :: Double
  , floorEffect :: Double
  , countyEffects :: [Double]
  }
  deriving (Show)

model :: [Row] -> Meas State
model rows = do
  mu0 <- sample (normal 1.0 1.5)
  tauRaw <- sample (gamma 2.0 1.0)
  sigmaRaw <- sample (gamma 2.0 1.0)
  betaFloor <- sample (normal 0.0 1.0)

  let tau = max 1e-6 tauRaw
      sigma = max 1e-6 sigmaRaw
      countyIds = [0 .. 9]

  effects <- mapM (\_ -> sample (normal mu0 tau)) countyIds

  let state =
        State
          { globalMean = mu0
          , countySd = tau
          , obsSd = sigma
          , floorEffect = betaFloor
          , countyEffects = effects
          }

  mapM_ (\r -> score (exp (logLikRow state r))) rows
  return state

safeCountyIndex :: Row -> Int
safeCountyIndex row =
  max 0 (round (county row))

floorIndicator :: Row -> Double
floorIndicator row =
  if basement row >= 0.5 then 1.0 else 0.0

logLikRow :: State -> Row -> Double
logLikRow state row =
  let yRaw = radon row
      y = max 1e-6 yRaw
      idxRaw = safeCountyIndex row
      nCounty = length (countyEffects state)
      idx =
        if nCounty <= 0
          then 0
          else max 0 (min (nCounty - 1) idxRaw)
      a = countyEffects state !! idx
      x = floorIndicator row
      mu = a + floorEffect state * x
      sigma = max 1e-6 (obsSd state)
   in normalLogPdf mu sigma (log y)